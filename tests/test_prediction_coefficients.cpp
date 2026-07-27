// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The invariants of the prediction coefficients, asserted on the coefficients
// themselves rather than through a mesh.
//
// Prediction is a property of the coefficients, so its invariants are checkable without
// building anything: no mesh, no field, no adaptation. That is why these tests are cheap
// enough to assert at every position class and every stencil shift, which a mesh-based
// test could only sample.
//
// Asserted here:
//   1. Reproduction   - the coefficients are exact on every monomial of degree <= 2r, in
//                       the cell-average sense, at every shift.
//   2. Conservativity - the two children's values average back to the parent exactly, at
//                       every shift.
//   3. Amplification  - sum |c| is bounded per shift, and the bound of the COMPOSED map
//                       does not grow with the level gap. The second half is the load
//                       bearing one: a bound that grew with the gap would mean the
//                       multiscale transform is not stable at the boundary.
//   4. Compatibility  - at shift 0 the coefficients reproduce interp_coeffs bit-exactly,
//                       so switching a consumer over cannot change an interior value.

#include <cmath>
#include <map>

#include <gtest/gtest.h>

#include <samurai/numeric/prediction.hpp>
#include <samurai/numeric/prediction_coefficients.hpp>

namespace samurai
{
    namespace
    {
        // Average of x^q over [lo, hi], in double. The tests below only ever compare it
        // against a combination of the same quantities, so the shared rounding cancels to
        // well within the tolerances used.
        double monomial_average(std::size_t q, double lo, double hi)
        {
            const double qq = static_cast<double>(q) + 1.;
            return (std::pow(hi, qq) - std::pow(lo, qq)) / (qq * (hi - lo));
        }

        // The parent cell is [0,1]; the stencil cell at offset s is [s, s+1]; the children
        // are the two halves of the parent.
        template <std::size_t radius>
        double reproduction_residual(std::size_t parity, int shift, std::size_t q)
        {
            const auto& p = prediction_coefficients<radius>(parity, shift);

            double lhs = 0.;
            for (std::size_t k = 0; k < p.size; ++k)
            {
                const double s = static_cast<double>(p.start + static_cast<int>(k));
                lhs += p.c[k] * monomial_average(q, s, s + 1.);
            }

            const double child_lo = (parity == 0) ? 0.0 : 0.5;
            const double rhs      = monomial_average(q, child_lo, child_lo + 0.5);
            return lhs - rhs;
        }

        template <std::size_t radius>
        double l1_mass(std::size_t parity, int shift)
        {
            const auto& p = prediction_coefficients<radius>(parity, shift);
            double s      = 0.;
            for (std::size_t k = 0; k < p.size; ++k)
            {
                s += std::abs(p.c[k]);
            }
            return s;
        }

        // ------------------------------------------------------------------
        // The composed map: prediction applied `gap` times in a row, expressed back in
        // the units of the coarsest level. Built exactly as reconstruction.hpp builds its
        // prediction maps, by recursion on the gap, so the bound measured here is the one
        // an actual multi-level reconstruction pays.
        //
        // `shift_of` decides the stencil at an absolute index, which is what makes this a
        // boundary measurement rather than an interior one: with a boundary at index 0 the
        // stencil clamps inward exactly where it has to.
        // ------------------------------------------------------------------
        using Map = std::map<long, double>;

        template <std::size_t radius, class ShiftOf>
        const Map& composed_map(long gap, long index, ShiftOf&& shift_of, std::map<std::pair<long, long>, Map>& cache)
        {
            auto key = std::make_pair(gap, index);
            auto it  = cache.find(key);
            if (it != cache.end())
            {
                return it->second;
            }

            if (gap == 0)
            {
                return cache[key] = Map{
                           {index, 1.}
                };
            }

            const long parent     = index >> 1; // arithmetic shift: floors, including for negatives
            const std::size_t par = static_cast<std::size_t>(index & 1);
            const auto& p         = prediction_coefficients<radius>(par, shift_of(parent));

            Map out;
            for (std::size_t k = 0; k < p.size; ++k)
            {
                if (p.c[k] == 0.)
                {
                    continue;
                }
                const long src = parent + p.start + static_cast<int>(k);
                for (const auto& kv : composed_map<radius>(gap - 1, src, shift_of, cache))
                {
                    out[kv.first] += p.c[k] * kv.second;
                }
            }
            return cache[key] = out;
        }

        // Worst sum |c| over the children of the coarse cell `p`, at level gap `gap`.
        template <std::size_t radius, class ShiftOf>
        double composed_l1(long gap, long p, ShiftOf&& shift_of)
        {
            std::map<std::pair<long, long>, Map> cache;
            const long nb = 1L << gap;
            double worst  = 0.;
            for (long ii = 0; ii < nb; ++ii)
            {
                double s = 0.;
                for (const auto& kv : composed_map<radius>(gap, p * nb + ii, shift_of, cache))
                {
                    s += std::abs(kv.second);
                }
                worst = std::max(worst, s);
            }
            return worst;
        }

        // Interior: never clamp.
        auto interior_shift()
        {
            return [](long)
            {
                return 0;
            };
        }

        // A boundary at index 0: cells below it do not exist at any level, so the stencil
        // clamps inward for the first `radius` cells. Nothing bounds the domain above, so
        // avail_high is passed at the cap of 2 * radius - passing `radius` there would make
        // prediction_shift report that the stencil does not fit, since a side short by
        // `radius` needs 2 * radius available opposite it.
        template <std::size_t radius>
        auto left_boundary_shift()
        {
            return [](long parent)
            {
                const int r         = static_cast<int>(radius);
                const int avail_low = (parent >= static_cast<long>(radius)) ? r : static_cast<int>(parent);
                const auto s        = prediction_shift(radius, avail_low, 2 * r);
                EXPECT_TRUE(s.fits) << "the stencil must fit at every position of this probe, parent=" << parent;
                return s.shift;
            };
        }

        template <std::size_t radius>
        void check_reproduction_and_conservativity()
        {
            const int r = static_cast<int>(radius);
            for (int shift = -r; shift <= r; ++shift)
            {
                for (std::size_t parity = 0; parity < 2; ++parity)
                {
                    // 1. exact on every monomial of degree <= 2r
                    for (std::size_t q = 0; q <= 2 * radius; ++q)
                    {
                        EXPECT_NEAR(reproduction_residual<radius>(parity, shift, q), 0., 1e-10)
                            << "radius=" << radius << " shift=" << shift << " parity=" << parity << " degree=" << q;
                    }
                }

                // 2. the children average back to the parent, as an identity between
                // coefficient vectors and not merely for particular data
                const auto& lo = prediction_coefficients<radius>(0, shift);
                const auto& hi = prediction_coefficients<radius>(1, shift);
                for (std::size_t k = 0; k < lo.size; ++k)
                {
                    const double mean     = 0.5 * (lo.c[k] + hi.c[k]);
                    const double expected = (lo.start + static_cast<int>(k) == 0) ? 1. : 0.;
                    EXPECT_NEAR(mean, expected, 1e-12) << "radius=" << radius << " shift=" << shift << " k=" << k;
                }
            }
        }
    }

    //-------------------------------------------------------------------------
    // 4. Bit-exact agreement with interp_coeffs at shift 0.
    //
    // This is what lets a consumer be switched over without changing any interior value,
    // so the acceptance bar for the boundary work can be "interior bit-identical" rather
    // than "interior within tolerance". Note EXPECT_DOUBLE_EQ, not EXPECT_NEAR.
    //-------------------------------------------------------------------------
    template <std::size_t radius>
    void check_matches_interp_coeffs()
    {
        constexpr std::size_t order = 2 * radius + 1;

        // interp_coeffs takes the sign of the child: +1 for the low child, -1 for the high.
        const auto legacy_lo = interp_coeffs<order>(1.);
        const auto legacy_hi = interp_coeffs<order>(-1.);

        const auto& lo = prediction_coefficients<radius>(0, 0);
        const auto& hi = prediction_coefficients<radius>(1, 0);

        EXPECT_EQ(lo.start, -static_cast<int>(radius));
        EXPECT_EQ(hi.start, -static_cast<int>(radius));

        for (std::size_t k = 0; k < order; ++k)
        {
            EXPECT_DOUBLE_EQ(lo.c[k], legacy_lo[k]) << "radius=" << radius << " low child, k=" << k;
            EXPECT_DOUBLE_EQ(hi.c[k], legacy_hi[k]) << "radius=" << radius << " high child, k=" << k;
        }
    }

    TEST(prediction_coefficients, matches_interp_coeffs_bit_exactly)
    {
        check_matches_interp_coeffs<1>();
        check_matches_interp_coeffs<2>();
        check_matches_interp_coeffs<3>();
        check_matches_interp_coeffs<4>();
        check_matches_interp_coeffs<5>();
    }

    //-------------------------------------------------------------------------
    // 1 and 2. Reproduction and conservativity, at every shift.
    //-------------------------------------------------------------------------
    TEST(prediction_coefficients, reproduction_and_conservativity_radius1)
    {
        check_reproduction_and_conservativity<1>();
    }

    TEST(prediction_coefficients, reproduction_and_conservativity_radius2)
    {
        check_reproduction_and_conservativity<2>();
    }

    TEST(prediction_coefficients, reproduction_and_conservativity_radius3)
    {
        check_reproduction_and_conservativity<3>();
    }

    // A negative control: the coefficients must NOT be exact one degree above 2r, else the
    // order is higher than claimed and the tests above are not pinning anything.
    TEST(prediction_coefficients, not_exact_above_2r)
    {
        EXPECT_GT(std::abs(reproduction_residual<1>(0, 0, 3)), 1e-6);
        EXPECT_GT(std::abs(reproduction_residual<1>(0, 1, 3)), 1e-6);
        EXPECT_GT(std::abs(reproduction_residual<2>(0, 0, 5)), 1e-6);
        EXPECT_GT(std::abs(reproduction_residual<2>(0, 2, 5)), 1e-6);
    }

    //-------------------------------------------------------------------------
    // 3. Amplification: bounded per shift, and NOT growing with the level gap.
    //-------------------------------------------------------------------------
    TEST(prediction_coefficients, one_step_l1_mass)
    {
        // Centred, and the fully one-sided stencil at each radius. The one-sided family
        // legitimately carries more l1 mass than the centred one; what matters is that it
        // is bounded, and that the composed bound below does not grow.
        EXPECT_NEAR(l1_mass<1>(0, 0), 1.25, 1e-12);
        EXPECT_NEAR(l1_mass<1>(0, 1), 2.0, 1e-12);
        EXPECT_NEAR(l1_mass<2>(0, 0), 89. / 64., 1e-12);
        EXPECT_NEAR(l1_mass<2>(0, 2), 3.5, 1e-12);
        EXPECT_NEAR(l1_mass<3>(0, 0), 381. / 256., 1e-12);
        EXPECT_NEAR(l1_mass<3>(0, 3), 55. / 8., 1e-12);
    }

    TEST(prediction_coefficients, composed_amplification_does_not_grow_with_the_level_gap)
    {
        // Interior: saturates at 4/3 in 1D.
        double prev_interior = 0.;
        for (long gap = 1; gap <= 12; ++gap)
        {
            const double v = composed_l1<1>(gap, 8, interior_shift());
            EXPECT_LT(v, 1.4) << "interior, gap=" << gap;
            EXPECT_GE(v, prev_interior - 1e-12) << "interior, gap=" << gap; // monotone up to roundoff
            prev_interior = v;
        }
        EXPECT_NEAR(composed_l1<1>(12, 8, interior_shift()), 4. / 3., 1e-3);

        // At the boundary: saturates at 10/3, i.e. 2.5x the interior. Bounded, and flat in
        // the gap, which is the property the multiscale stability argument needs and the
        // one the literature does not establish for cell averages.
        double prev_boundary = 0.;
        for (long gap = 1; gap <= 12; ++gap)
        {
            const double v = composed_l1<1>(gap, 0, left_boundary_shift<1>());
            EXPECT_LT(v, 3.4) << "boundary, gap=" << gap;
            EXPECT_GE(v, prev_boundary - 1e-12) << "boundary, gap=" << gap;
            prev_boundary = v;
        }
        EXPECT_NEAR(composed_l1<1>(12, 0, left_boundary_shift<1>()), 10. / 3., 1e-2);

        // And a cell 2r away from the boundary is already indistinguishable from the
        // interior: the clamping never reaches it, so only the first 2r cells deviate.
        EXPECT_DOUBLE_EQ(composed_l1<1>(10, 2, left_boundary_shift<1>()), composed_l1<1>(10, 8, interior_shift()));
    }

    //-------------------------------------------------------------------------
    // The shift classifier, which decides WHICH stencil to use. Pure integer arithmetic on
    // the geometry of the domain, so it is partition independent by construction: it never
    // looks at what a rank happens to hold.
    //-------------------------------------------------------------------------
    TEST(prediction_shift, centred_when_the_stencil_fits)
    {
        for (std::size_t r = 1; r <= 4; ++r)
        {
            const auto s = prediction_shift(r, static_cast<int>(r), static_cast<int>(r));
            EXPECT_TRUE(s.fits);
            EXPECT_EQ(s.shift, 0) << "radius=" << r;
        }
    }

    TEST(prediction_shift, clamps_inward_by_the_deficit)
    {
        // radius 2, low side short by 1 then by 2: shift up by exactly the deficit.
        EXPECT_EQ(prediction_shift(2, 1, 5).shift, 1);
        EXPECT_EQ(prediction_shift(2, 0, 5).shift, 2);
        // high side, mirrored
        EXPECT_EQ(prediction_shift(2, 5, 1).shift, -1);
        EXPECT_EQ(prediction_shift(2, 5, 0).shift, -2);
        // never more than the radius
        EXPECT_LE(std::abs(prediction_shift(3, 0, 9).shift), 3);
    }

    TEST(prediction_shift, reports_when_no_shift_fits)
    {
        // The band must hold 2r+1 cells: avail_low + avail_high >= 2r.
        EXPECT_TRUE(prediction_shift(2, 0, 4).fits);
        EXPECT_FALSE(prediction_shift(2, 0, 3).fits);
        EXPECT_FALSE(prediction_shift(3, 2, 3).fits);
        EXPECT_TRUE(prediction_shift(3, 3, 3).fits);

        // radius 1 needs only 2 cells beyond the parent, which is why it is the radius
        // that fits everywhere in practice today.
        EXPECT_TRUE(prediction_shift(1, 0, 2).fits);
        EXPECT_FALSE(prediction_shift(1, 0, 1).fits);
    }
}
