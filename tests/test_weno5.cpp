// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// compute_weno5_flux on vector values.
//
// Vector values are handled one component at a time with the scalar formula. Each
// component of the vector flux must then equal the scalar flux of that component, for
// the two flux containers (xtensor and StdArrayWrapper) and the two stencil containers
// the convection schemes use. The scalar flux itself is checked against the formula of
// Jiang and Shu written out below.

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>

#include <samurai/schemes/fv.hpp>

namespace samurai
{
    namespace
    {
        constexpr std::size_t n_comp = 3;

        // Stencil value k of component c. Each component has its own profile, so that
        // mixing two components up changes the flux.
        double stencil_value(std::size_t k, std::size_t c)
        {
            const double x = static_cast<double>(k);
            switch (c)
            {
                case 0:
                    return std::sin(0.7 * x); // smooth
                case 1:
                    return k < 3 ? 1. : -2.; // jump
                default:
                    return 0.25 * x * x - x; // changes sign
            }
        }

        template <class Stencil>
        Stencil make_stencil()
        {
            Stencil f;
            for (std::size_t k = 0; k < 5; ++k)
            {
                for (std::size_t c = 0; c < n_comp; ++c)
                {
                    f[k][c] = stencil_value(k, c);
                }
            }
            return f;
        }

        double scalar_flux(std::size_t c)
        {
            std::array<double, 5> f;
            for (std::size_t k = 0; k < 5; ++k)
            {
                f[k] = stencil_value(k, c);
            }
            double flux = std::numeric_limits<double>::quiet_NaN();
            compute_weno5_flux(flux, f);
            return flux;
        }

        // A flux full of NaN, so that a component left unset is caught.
        template <class Value>
        Value make_unset_flux()
        {
            Value flux;
            for (std::size_t c = 0; c < n_comp; ++c)
            {
                flux[c] = std::numeric_limits<double>::quiet_NaN();
            }
            return flux;
        }

        double square(double x)
        {
            return x * x;
        }

        // WENO5 flux of component c, from 'Efficient implementation of Weighted ENO
        // schemes', Jiang and Shu, 1996: (2.8), (3.2)-(3.4), (2.15)-(2.16) and (2.10).
        double jiang_shu_flux(std::size_t c)
        {
            std::array<double, 5> f;
            for (std::size_t k = 0; k < 5; ++k)
            {
                f[k] = stencil_value(k, c);
            }

            const std::array<double, 3> q = {
                (2 * f[0] - 7 * f[1] + 11 * f[2]) / 6,
                (-f[1] + 5 * f[2] + 2 * f[3]) / 6,
                (2 * f[2] + 5 * f[3] - f[4]) / 6,
            };
            const std::array<double, 3> is = {
                13. / 12 * square(f[0] - 2 * f[1] + f[2]) + square(f[0] - 4 * f[1] + 3 * f[2]) / 4,
                13. / 12 * square(f[1] - 2 * f[2] + f[3]) + square(f[1] - f[3]) / 4,
                13. / 12 * square(f[2] - 2 * f[3] + f[4]) + square(3 * f[2] - 4 * f[3] + f[4]) / 4,
            };
            const std::array<double, 3> linear_weight = {0.1, 0.6, 0.3};

            double sum_alphas = 0;
            double flux       = 0;
            for (std::size_t r = 0; r < 3; ++r)
            {
                const double alpha = linear_weight[r] / square(1e-6 + is[r]);
                sum_alphas += alpha;
                flux += alpha * q[r];
            }
            return flux / sum_alphas;
        }

        template <class Stencil>
        void expect_scalar_flux_per_component()
        {
            using value_t = typename Stencil::value_type;

            const auto f = make_stencil<Stencil>();
            auto flux    = make_unset_flux<value_t>();

            compute_weno5_flux(flux, f);

            for (std::size_t c = 0; c < n_comp; ++c)
            {
                EXPECT_DOUBLE_EQ(flux[c], scalar_flux(c)) << "component " << c;
            }
        }
    }

    TEST(weno5_scalar_flux, matches_jiang_shu_formula)
    {
        for (std::size_t c = 0; c < n_comp; ++c)
        {
            const double expected = jiang_shu_flux(c);
            EXPECT_NEAR(scalar_flux(c), expected, 1e-13 * std::abs(expected)) << "profile " << c;
        }
    }

    template <class Value>
    class weno5_vector_flux : public ::testing::Test
    {
    };

    using weno5_flux_values = ::testing::Types<xtensor_static_array<double, n_comp>, StdArrayWrapper<double, n_comp>>;
    TYPED_TEST_SUITE(weno5_vector_flux, weno5_flux_values, );

    // Stencil container of the nonlinear convection.
    TYPED_TEST(weno5_vector_flux, std_array_stencil_matches_scalar_flux)
    {
        expect_scalar_flux_per_component<std::array<TypeParam, 5>>();
    }

    // Stencil container of the linear convection.
    TYPED_TEST(weno5_vector_flux, static_array_stencil_matches_scalar_flux)
    {
        expect_scalar_flux_per_component<Array<TypeParam, 5>>();
    }

    // The three candidate stencils of WENO5 are exact on linear data, whatever the
    // weights: the flux is the value at the face, 2.5 cells after the first cell center.
    TYPED_TEST(weno5_vector_flux, linear_data_is_exact_at_the_face)
    {
        const std::array<double, n_comp> slope  = {1., -0.5, 3.};
        const std::array<double, n_comp> origin = {0., 2., -7.};

        std::array<TypeParam, 5> f;
        for (std::size_t k = 0; k < 5; ++k)
        {
            for (std::size_t c = 0; c < n_comp; ++c)
            {
                f[k][c] = slope[c] * static_cast<double>(k) + origin[c];
            }
        }

        auto flux = make_unset_flux<TypeParam>();
        compute_weno5_flux(flux, f);

        for (std::size_t c = 0; c < n_comp; ++c)
        {
            EXPECT_NEAR(flux[c], slope[c] * 2.5 + origin[c], 1e-13) << "component " << c;
        }
    }
}
