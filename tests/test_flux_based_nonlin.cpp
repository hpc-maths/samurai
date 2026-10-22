// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// NonLinear flux-based schemes with an odd stencil size.
//
// The fluxes at finer levels (--finer-level-flux, or the finer_level_flux()
// setting of a scheme) are not implemented for odd stencil sizes, except when
// the prediction stencil radius is 0 and the stencil has at most 4 cells. Such a
// scheme must still compile and run without that option, and throw when it is
// enabled.

#include <array>
#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include <samurai/arguments.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

namespace samurai
{
    namespace
    {
        // Upwind flux a[d] * u(left cell of the face) on the default stencil of `size` cells.
        template <std::size_t size, class Field>
        auto make_nonlinear_upwind(const std::array<double, Field::dim>& a)
        {
            static constexpr std::size_t dim = Field::dim;
            using cfg                        = FluxConfig<SchemeType::NonLinear, size, Field, Field>;
            // The default stencil starts at -size / 2 + 1: the left cell of the face has the index size / 2 - 1.
            static constexpr std::size_t left = size / 2 - 1;

            FluxDefinition<cfg> flux;
            static_for<0, dim>::apply(
                [&](auto integral_constant_d)
                {
                    static constexpr std::size_t d = decltype(integral_constant_d)::value;

                    double ad                  = a[d];
                    flux[d].cons_flux_function = [ad](FluxValue<cfg>& f, const StencilData<cfg>&, const StencilValues<cfg>& u)
                    {
                        f = ad * u[left];
                    };
                });
            return make_flux_based_scheme(flux);
        }

        // A periodic mesh of [0, 1]^dim, from level 2 to level 6.
        template <std::size_t dim>
        auto periodic_mesh()
        {
            using Box     = samurai::Box<double, dim>;
            using point_t = typename Box::point_t;
            point_t corner1, corner2;
            corner1.fill(0.);
            corner2.fill(1.);
            auto cfg = mesh_config<dim>().min_level(2).max_level(6).periodic(true).max_stencil_size(6);
            return mra::make_mesh(Box(corner1, corner2), cfg);
        }

        // Sets u to a Gaussian and adapts its mesh to it, which creates level jumps.
        template <class Field>
        void set_gaussian_and_adapt(Field& u)
        {
            static constexpr std::size_t dim = Field::dim;
            for_each_cell(u.mesh(),
                          [&](auto& cell)
                          {
                              double r2 = 0;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r2 += (cell.center(d) - 0.5) * (cell.center(d) - 0.5);
                              }
                              u[cell] = std::exp(-100 * r2);
                          });
            auto adapt = make_MRAdapt(u);
            adapt(mra_config().epsilon(1e-3));
        }

        template <std::size_t dim>
        std::array<double, dim> velocity()
        {
            std::array<double, dim> a;
            for (std::size_t d = 0; d < dim; ++d)
            {
                a[d] = 1. / static_cast<double>(d + 1);
            }
            return a;
        }

        template <class Field>
        double max_difference(const Field& v, const Field& w)
        {
            double diff = 0;
            for_each_cell(v.mesh(),
                          [&](auto& cell)
                          {
                              diff = std::max(diff, std::abs(v[cell] - w[cell]));
                          });
            return diff;
        }

        // The integral of v over the mesh: zero for the divergence of a flux on a periodic mesh.
        template <class Field>
        double integral(const Field& v)
        {
            double sum = 0;
            for_each_cell(v.mesh(),
                          [&](auto& cell)
                          {
                              sum += v[cell] * std::pow(cell.length, static_cast<double>(Field::dim));
                          });
            return sum;
        }

        // Restores args::finer_level_flux when a test ends.
        struct FinerLevelFluxGuard
        {
            int saved = args::finer_level_flux;

            FinerLevelFluxGuard() = default;

            FinerLevelFluxGuard(const FinerLevelFluxGuard&)            = delete;
            FinerLevelFluxGuard& operator=(const FinerLevelFluxGuard&) = delete;

            ~FinerLevelFluxGuard()
            {
                args::finer_level_flux = saved;
            }
        };

        constexpr double tol = 1e-12;
    }

    template <class T>
    class flux_based_nonlin_odd_stencil : public ::testing::Test
    {
    };

    using dims = ::testing::Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>>;
    TYPED_TEST_SUITE(flux_based_nonlin_odd_stencil, dims, );

    // Without fluxes at finer levels, the stencils of 3 and 5 cells give the linear upwind scheme.
    TYPED_TEST(flux_based_nonlin_odd_stencil, matches_linear_upwind)
    {
        static constexpr std::size_t dim = TypeParam::value;
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 0;

        auto mesh = periodic_mesh<dim>();
        auto u    = make_scalar_field<double>("u", mesh);
        set_gaussian_and_adapt(u);
        using Field = decltype(u);
        auto a      = velocity<dim>();
        VelocityVector<dim> av;
        for (std::size_t d = 0; d < dim; ++d)
        {
            av[d] = a[d];
        }

        auto linear = make_convection_upwind<Field>(av);
        auto size_2 = make_nonlinear_upwind<2, Field>(a);
        auto size_3 = make_nonlinear_upwind<3, Field>(a);
        auto size_5 = make_nonlinear_upwind<5, Field>(a);

        auto v_linear = make_scalar_field<double>("v_linear", u.mesh());
        auto v_2      = make_scalar_field<double>("v_2", u.mesh());
        auto v_3      = make_scalar_field<double>("v_3", u.mesh());
        auto v_5      = make_scalar_field<double>("v_5", u.mesh());
        v_linear      = linear(u);
        v_2           = size_2(u);
        v_3           = size_3(u);
        v_5           = size_5(u);

        EXPECT_GT(u.mesh().max_level(), u.mesh().min_level()); // the mesh has level jumps
        EXPECT_LT(max_difference(v_2, v_linear), tol);
        EXPECT_LT(max_difference(v_3, v_linear), tol);
        EXPECT_LT(max_difference(v_5, v_linear), tol);
    }

    // With fluxes at finer levels, an odd stencil size throws, whether the option is global or set on the scheme.
    TYPED_TEST(flux_based_nonlin_odd_stencil, finer_level_flux_throws)
    {
        static constexpr std::size_t dim = TypeParam::value;
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 0;

        auto mesh = periodic_mesh<dim>();
        auto u    = make_scalar_field<double>("u", mesh);
        set_gaussian_and_adapt(u);
        using Field = decltype(u);
        auto a      = velocity<dim>();

        auto size_3 = make_nonlinear_upwind<3, Field>(a);
        auto size_5 = make_nonlinear_upwind<5, Field>(a);

        size_3.finer_level_flux() = 1;
        EXPECT_THROW(size_3(u), std::runtime_error);
        size_3.finer_level_flux() = 0;
        EXPECT_NO_THROW(size_3(u));

        args::finer_level_flux = 1;
        EXPECT_THROW(size_5(u), std::runtime_error);
        args::finer_level_flux = -1;
        EXPECT_THROW(size_5(u), std::runtime_error);
    }

    // An even stencil size keeps its fluxes at finer levels: the stencils of 2 and 4 cells agree,
    // and the scheme is conservative.
    TYPED_TEST(flux_based_nonlin_odd_stencil, even_stencil_with_finer_level_flux)
    {
        static constexpr std::size_t dim = TypeParam::value;
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 1;

        auto mesh = periodic_mesh<dim>();
        auto u    = make_scalar_field<double>("u", mesh);
        set_gaussian_and_adapt(u);
        using Field = decltype(u);
        auto a      = velocity<dim>();

        auto size_2 = make_nonlinear_upwind<2, Field>(a);
        auto size_4 = make_nonlinear_upwind<4, Field>(a);

        auto v_2 = make_scalar_field<double>("v_2", u.mesh());
        auto v_4 = make_scalar_field<double>("v_4", u.mesh());
        EXPECT_NO_THROW(v_2 = size_2(u));
        EXPECT_NO_THROW(v_4 = size_4(u));

        EXPECT_LT(max_difference(v_2, v_4), tol);
        EXPECT_LT(std::abs(integral(v_2)), tol);
    }
}
