// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// mra_config::min_level_in(region, level): on the region, every cell is at a level of at
// least `level`, whatever the multiresolution criterion says there. A guarantee, so the tests
// ask for it on a field the criterion would coarsen everywhere, and where the constraint is
// the only reason a fine cell exists.

#include <cmath>

#include <gtest/gtest.h>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

namespace samurai
{
    namespace
    {
        template <std::size_t dim>
        auto uniform_field_mesh(std::size_t min_level, std::size_t max_level)
        {
            auto cfg  = mesh_config<dim>().min_level(min_level).max_level(max_level);
            auto mesh = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, cfg);
            return mesh;
        }

        /// The coarsest level found among the cells whose centre lies in @a region, and how many there are.
        template <class Mesh, std::size_t dim>
        std::pair<std::size_t, std::size_t> coarsest_level_in(const Mesh& mesh, const Box<double, dim>& region)
        {
            std::size_t coarsest = mesh.max_level() + 1;
            std::size_t count    = 0;
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const auto c = cell.center();
                              bool inside  = true;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  inside = inside && c[d] >= region.min_corner()[d] && c[d] <= region.max_corner()[d];
                              }
                              if (inside)
                              {
                                  ++count;
                                  coarsest = std::min(coarsest, cell.level);
                              }
                          });
            return {coarsest, count};
        }
    }

    TEST(level_constraint, a_constant_field_is_kept_fine_on_the_region_only)
    {
        // The criterion coarsens a constant field down to min_level everywhere; the constraint
        // holds a strip along the left boundary at max_level, and nothing else.
        constexpr std::size_t dim = 2;
        auto mesh                 = uniform_field_mesh<dim>(2, 6);
        auto u                    = make_scalar_field<double>("u", mesh);
        u.fill(1.);
        make_bc<Dirichlet<1>>(u, 1.);

        const Box<double, dim> strip({0., 0.}, {0.1, 1.});
        auto cfg = mra_config().epsilon(1e-3).min_level_in(strip, 6);
        make_MRAdapt(u)(cfg);

        const auto [coarsest_in, count_in] = coarsest_level_in(mesh, strip);
        EXPECT_GT(count_in, 0u);
        EXPECT_EQ(coarsest_in, 6u);

        const Box<double, dim> far({0.5, 0.}, {1., 1.});
        const auto [coarsest_far, count_far] = coarsest_level_in(mesh, far);
        EXPECT_GT(count_far, 0u);
        EXPECT_EQ(coarsest_far, 2u);
    }

    TEST(level_constraint, an_intermediate_level_is_created_where_the_mesh_was_coarser)
    {
        // Starting from a mesh already coarsened to min_level, the constraint *creates* the
        // refinement - which the flag it replaces could not do - and only up to the level asked.
        constexpr std::size_t dim = 2;
        auto mesh                 = uniform_field_mesh<dim>(2, 6);
        auto u                    = make_scalar_field<double>("u", mesh);
        u.fill(1.);
        make_bc<Dirichlet<1>>(u, 1.);

        make_MRAdapt(u)(mra_config().epsilon(1e-3));
        ASSERT_EQ(mesh.min_level(), 2u);
        {
            const auto [coarsest, count] = coarsest_level_in(mesh, Box<double, dim>({0., 0.}, {1., 1.}));
            ASSERT_GT(count, 0u);
            ASSERT_EQ(coarsest, 2u);
        }

        const Box<double, dim> patch({0.4, 0.4}, {0.6, 0.6});
        make_MRAdapt(u)(mra_config().epsilon(1e-3).min_level_in(patch, 4));

        const auto [coarsest_in, count_in] = coarsest_level_in(mesh, patch);
        EXPECT_GT(count_in, 0u);
        EXPECT_EQ(coarsest_in, 4u);

        // Not finer than asked: the criterion would coarsen it, the constraint stops at 4.
        std::size_t finest_in = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          const auto c = cell.center();
                          if (c[0] >= 0.4 && c[0] <= 0.6 && c[1] >= 0.4 && c[1] <= 0.6)
                          {
                              finest_in = std::max(finest_in, cell.level);
                          }
                      });
        EXPECT_EQ(finest_in, 4u);
    }

    TEST(level_constraint, the_constraint_survives_a_moving_solution)
    {
        // Several adaptations of a field whose feature moves away from the region: the region
        // stays at its level while the rest of the mesh follows the feature.
        constexpr std::size_t dim = 2;
        auto mesh                 = uniform_field_mesh<dim>(2, 6);
        auto u                    = make_scalar_field<double>("u", mesh);
        make_bc<Dirichlet<1>>(u, 0.);
        auto adapt = make_MRAdapt(u);

        const Box<double, dim> corner({0.9, 0.9}, {1., 1.});
        for (std::size_t step = 0; step < 4; ++step)
        {
            const double centre = 0.2 + 0.15 * static_cast<double>(step);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const auto c   = cell.center();
                              const double r = std::hypot(c[0] - centre, c[1] - centre);
                              u[cell]        = r < 0.1 ? 1. : 0.;
                          });
            adapt(mra_config().epsilon(1e-3).min_level_in(corner, 5));

            const auto [coarsest, count] = coarsest_level_in(mesh, corner);
            EXPECT_GT(count, 0u) << "step " << step;
            EXPECT_GE(coarsest, 5u) << "step " << step;
        }
    }

    TEST(level_constraint, a_region_of_the_wrong_dimension_is_refused)
    {
        constexpr std::size_t dim = 2;
        auto mesh                 = uniform_field_mesh<dim>(2, 4);
        auto u                    = make_scalar_field<double>("u", mesh);
        u.fill(1.);
        make_bc<Dirichlet<1>>(u, 1.);

        auto cfg = mra_config().epsilon(1e-3).min_level_in(Box<double, 3>({0., 0., 0.}, {1., 1., 1.}), 4);
        EXPECT_THROW(make_MRAdapt(u)(cfg), std::invalid_argument);
    }
}
