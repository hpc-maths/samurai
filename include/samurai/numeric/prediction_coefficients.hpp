// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <unordered_map>

#include "../utils.hpp"

/**
 * @file prediction_coefficients.hpp
 *
 * The 1D prediction coefficients, and the shift a stencil takes near a boundary.
 *
 * Prediction interpolates a coarse cell's value onto its children. In dim > 1 it is a
 * tensor product of this 1D family, one factor per direction, so this file holds the
 * whole of what its consumers share: each of them builds its own product from it and
 * keeps its own loop structure.
 *
 * A stencil of prediction stencil radius @c r reads @c 2r+1 consecutive coarse cells.
 * Away from a boundary it is centred on the parent. Near one it is **shifted inward**
 * so that it reads only cells that exist, and the coefficients are re-solved for that
 * shift. @c shift == 0 is the centred family and reproduces
 * @c interp_coeffs<2r+1> (numeric/prediction.hpp) bit-exactly.
 *
 * Coefficients come from the **cell-average** moment conditions: for every monomial of
 * degree <= @c 2r, the combination of the stencil cells' averages must equal the
 * child's average. That is a @c (2r+1)x(2r+1) system, solved here in exact rational
 * arithmetic and rounded to double once at the end, so a dyadic value - which is every
 * value of the centred family - comes out bit-exact rather than merely close.
 *
 * Two things are deliberately kept apart, because conflating them is what would make a
 * distributed run depend on its partition:
 *   - @ref prediction_shift decides *which* stencil to use, from the geometry of the
 *     domain alone;
 *   - whether the cells that stencil names are present locally is a separate, halo
 *     question, and not this file's business.
 */

namespace samurai
{
    /**
     * The 1D prediction coefficients of one child, as a stencil start plus @c 2r+1
     * weights: the child's value is @c sum_k c[k] * u(parent + start + k).
     */
    template <std::size_t radius>
    struct PredictionCoefficients
    {
        static constexpr std::size_t size = 2 * radius + 1;

        int start = -static_cast<int>(radius); ///< offset of @c c[0] from the parent cell
        std::array<double, size> c{};
    };

    namespace detail
    {
        /**
         * Minimal exact rational, enough to solve the moment system. Normalised after
         * every operation so the magnitudes stay small: for @c radius <= 5 they remain
         * far inside @c std::int64_t, which the bit-exactness tests against
         * @c interp_coeffs would catch if it ever stopped being true.
         */
        struct Rational
        {
            std::int64_t n = 0;
            std::int64_t d = 1;

            constexpr Rational() = default;

            constexpr explicit Rational(std::int64_t num, std::int64_t den = 1)
                : n(num)
                , d(den)
            {
                normalise();
            }

            constexpr void normalise()
            {
                if (d < 0)
                {
                    n = -n;
                    d = -d;
                }
                const std::int64_t g = std::gcd(n < 0 ? -n : n, d);
                if (g > 1)
                {
                    n /= g;
                    d /= g;
                }
                if (n == 0)
                {
                    d = 1;
                }
            }

            constexpr Rational operator+(const Rational& o) const
            {
                return Rational{n * o.d + o.n * d, d * o.d};
            }

            constexpr Rational operator-(const Rational& o) const
            {
                return Rational{n * o.d - o.n * d, d * o.d};
            }

            constexpr Rational operator*(const Rational& o) const
            {
                return Rational{n * o.n, d * o.d};
            }

            constexpr Rational operator/(const Rational& o) const
            {
                return Rational{n * o.d, d * o.n};
            }

            constexpr bool is_zero() const
            {
                return n == 0;
            }

            constexpr double to_double() const
            {
                return static_cast<double>(n) / static_cast<double>(d);
            }
        };

        constexpr Rational pow(Rational x, std::size_t e)
        {
            Rational r{1};
            for (std::size_t k = 0; k < e; ++k)
            {
                r = r * x;
            }
            return r;
        }

        /// Average of @c x^q over @c [lo, hi]. The moment the system is posed on.
        constexpr Rational monomial_average(std::size_t q, Rational lo, Rational hi)
        {
            const Rational num = pow(hi, q + 1) - pow(lo, q + 1);
            return num / (Rational{static_cast<std::int64_t>(q) + 1} * (hi - lo));
        }

        /**
         * Solve the moment system for one child. The parent cell is @c [0,1], the
         * stencil cell at offset @c s is @c [s, s+1], and the child is the low or high
         * half of the parent. Gauss-Jordan with partial pivoting on exact rationals, so
         * the result is the exact solution of an exactly-posed system.
         */
        template <std::size_t radius>
        std::array<double, 2 * radius + 1> solve_moment_system(std::size_t parity, int shift)
        {
            constexpr std::size_t n = 2 * radius + 1;
            const int start         = -static_cast<int>(radius) + shift;

            const Rational child_lo = (parity == 0) ? Rational{0} : Rational{1, 2};
            const Rational child_hi = (parity == 0) ? Rational{1, 2} : Rational{1};

            // row q: sum_k a[k] * avg(x^q over stencil cell k) = avg(x^q over the child)
            std::array<std::array<Rational, n>, n> a{};
            std::array<Rational, n> rhs{};
            for (std::size_t q = 0; q < n; ++q)
            {
                for (std::size_t k = 0; k < n; ++k)
                {
                    const auto s = Rational{start + static_cast<int>(k)};
                    a[q][k]      = monomial_average(q, s, s + Rational{1});
                }
                rhs[q] = monomial_average(q, child_lo, child_hi);
            }

            // `col` and `row` would shadow samurai::col() / samurai::row()
            // (storage/xtensor/xtensor_static.hpp), so the indices are named for what they are.
            for (std::size_t pivot_col = 0; pivot_col < n; ++pivot_col)
            {
                std::size_t pivot_row = pivot_col;
                while (pivot_row < n && a[pivot_row][pivot_col].is_zero())
                {
                    ++pivot_row;
                }
                std::swap(a[pivot_col], a[pivot_row]);
                std::swap(rhs[pivot_col], rhs[pivot_row]);

                const Rational p = a[pivot_col][pivot_col];
                for (std::size_t k = 0; k < n; ++k)
                {
                    a[pivot_col][k] = a[pivot_col][k] / p;
                }
                rhs[pivot_col] = rhs[pivot_col] / p;

                for (std::size_t other = 0; other < n; ++other)
                {
                    if (other == pivot_col || a[other][pivot_col].is_zero())
                    {
                        continue;
                    }
                    const Rational f = a[other][pivot_col];
                    for (std::size_t k = 0; k < n; ++k)
                    {
                        a[other][k] = a[other][k] - f * a[pivot_col][k];
                    }
                    rhs[other] = rhs[other] - f * rhs[pivot_col];
                }
            }

            std::array<double, n> out{};
            for (std::size_t k = 0; k < n; ++k)
            {
                out[k] = rhs[k].to_double();
            }
            return out;
        }
    }

    /**
     * The 1D prediction coefficients of the child of parity @a parity, for a stencil
     * shifted inward by @a shift cells.
     *
     * @param parity 0 for the child on the low side of the parent, 1 for the high side.
     *               Matches the low bit of the child's index.
     * @param shift  inward displacement of the stencil, in cells: positive moves it away
     *               from a low-side boundary, negative away from a high-side one, and 0
     *               is the centred stencil. @c |shift| <= radius.
     *
     * Memoised: the solve happens once per @c (radius, parity, shift), and there are at
     * most @c 2*(2*radius+1) of those.
     */
    template <std::size_t radius>
    const PredictionCoefficients<radius>& prediction_coefficients(std::size_t parity, int shift)
    {
        assert(parity < 2 && "prediction_coefficients: parity is 0 or 1");
        assert(std::abs(shift) <= static_cast<int>(radius) && "prediction_coefficients: |shift| must not exceed the stencil radius");

        static std::unordered_map<int, PredictionCoefficients<radius>> cache;

        const int key = 2 * shift + static_cast<int>(parity);
        auto it       = cache.find(key);
        if (it != cache.end())
        {
            return it->second;
        }

        PredictionCoefficients<radius> out;
        out.start = -static_cast<int>(radius) + shift;
        out.c     = detail::solve_moment_system<radius>(parity, shift);
        return cache.emplace(key, out).first->second;
    }

    /// What @ref prediction_shift concluded about a position.
    struct PredictionShift
    {
        int shift = 0;    ///< pass to @ref prediction_coefficients
        bool fits = true; ///< false when no shift makes the stencil fit
    };

    /**
     * The shift a prediction stencil must take at a given position.
     *
     * @param radius     prediction stencil radius
     * @param avail_low  how many cells are available below the parent, at its level
     * @param avail_high how many are available above it
     *
     * @c fits is false exactly when @c avail_low + avail_high < 2 * radius, i.e. when the
     * band available at that level is too narrow to hold @c 2r+1 cells however it is
     * shifted. That is the constructibility condition the mesh must guarantee, and a
     * caller is expected to report it loudly rather than silently degrade the order.
     *
     * @warning Either count may be **capped at @c 2 * radius**, never lower. A deficit on
     * one side is at most @c radius and is made up from the other side, so @c 2 * radius
     * is all the information the answer can use - but capping at @c radius instead would
     * make @c fits report false at a boundary where the stencil does fit, since a side
     * short by @c radius needs @c 2 * radius available opposite it.
     */
    constexpr PredictionShift prediction_shift(std::size_t radius, int avail_low, int avail_high)
    {
        const int r = static_cast<int>(radius);

        if (avail_low + avail_high < 2 * r)
        {
            return {0, false};
        }

        // A deficit on one side is made up by shifting towards the other, and at most one
        // of the two deficits can be positive once the stencil fits at all.
        const int deficit_low  = r - avail_low;
        const int deficit_high = r - avail_high;

        if (deficit_low > 0)
        {
            return {deficit_low, true};
        }
        if (deficit_high > 0)
        {
            return {-deficit_high, true};
        }
        return {0, true};
    }
}
