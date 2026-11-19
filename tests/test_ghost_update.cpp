#include <cmath>
#include <limits>
#include <ranges>
#include <string>

#include <gtest/gtest.h>

#include <samurai/algorithm/update_outer_ghost.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>

namespace samurai
{
    namespace
    {
        // Build a 3D adapted mesh and run update_ghost_mr on a field created after
        // adaptation. `boundary_touching` selects the refined region:
        //  - false: an interior blob that never reaches the domain boundary;
        //  - true : two fronts spanning the whole domain, reaching the corners.
        void run_ghost_update_3d(bool boundary_touching)
        {
            constexpr std::size_t dim = 3;

            auto config = mesh_config<dim>().min_level(4).max_level(6).max_stencil_size(2).disable_minimal_ghost_width();
            auto mesh   = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, config);

            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              auto c = cell.center();
                              if (boundary_touching)
                              {
                                  const std::size_t nb = 2;
                                  u[cell]              = 0;
                                  for (std::size_t i = 1; i <= nb; ++i)
                                  {
                                      u[cell] += std::tanh(1000 * std::abs(c[0] - static_cast<double>(i) / static_cast<double>(nb + 1)));
                                  }
                                  u[cell] -= static_cast<double>(nb);
                              }
                              else
                              {
                                  double r = 0;
                                  for (std::size_t d = 0; d < dim; ++d)
                                  {
                                      r += (c[d] - 0.3) * (c[d] - 0.3);
                                  }
                                  u[cell] = (r <= 0.04) ? 1. : 0.;
                              }
                          });
            make_bc<Dirichlet<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));

            auto v = make_scalar_field<double>("v", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              v[cell] = cell.center(0);
                          });
            make_bc<Dirichlet<1>>(v, 0.);
            update_ghost_mr(v);
        }

        enum class OuterBc
        {
            neumann,
            dirichlet,
            none
        };

        /**
         * Adapts a 2D mesh of [0,4]x[0,1] to a sequence of features that cross the walls and fill the corners
         * at varying sizes, so that the levels jump along the walls and the corners are refined at varying
         * depths. On each mesh, a constant field whose ghosts are all NaN goes through update_ghost_mr, and
         * every ghost the solver reads must then hold the constant:
         *  - the box of radius ghost_width around the cells (flux stencils, finer-level-flux predictions);
         *  - the box of radius prediction_stencil_radius around the parents and the grandparents of the cells
         *    (details, predictions).
         * Returns the number of ghosts that do not.
         */
        template <int prediction_stencil_radius>
        std::size_t count_wrong_outer_ghosts(OuterBc outer_bc)
        {
            constexpr std::size_t dim = 2;
            constexpr double value    = 3.;

            auto config = mesh_config<dim, prediction_stencil_radius>()
                              .min_level(4)
                              .max_level(7)
                              .max_stencil_size(4 * prediction_stencil_radius)
                              .disable_minimal_ghost_width();
            auto mesh       = mra::make_mesh(Box<double, dim>({0., 0.}, {4., 1.}), config);
            using mesh_id_t = typename decltype(mesh)::mesh_id_t;

            auto u    = make_scalar_field<double>("u", mesh);
            int step  = 0;
            auto init = [&]()
            {
                const double s = step;
                for_each_cell(mesh,
                              [&](auto& cell)
                              {
                                  auto c   = cell.center();
                                  double w = std::tanh(200 * (c[0] - 0.2 - 0.5 * s));
                                  w += std::tanh(200 * (std::hypot(c[0] - 0.5 - 0.4 * s, c[1]) - 0.2));
                                  w += std::tanh(200 * (std::hypot(c[0] - 2.5 + 0.3 * s, c[1] - 1.) - 0.15));
                                  const std::array<double, 4> xc = {0., 4., 0., 4.};
                                  const std::array<double, 4> yc = {0., 0., 1., 1.};
                                  for (std::size_t k = 0; k < 4; ++k)
                                  {
                                      const double rc = 0.02 + 0.05 * static_cast<double>((step + 3 * static_cast<int>(k)) % 6);
                                      w += std::tanh(300 * (std::hypot(c[0] - xc[k], c[1] - yc[k]) - rc));
                                  }
                                  u[cell] = w;
                              });
            };
            // Neumann, so that the walls do not get refined for their own sake.
            make_bc<Neumann<1>>(u, 0.);
            auto adapt = make_MRAdapt(u);

            std::size_t n_wrong = 0;
            for (step = 0; step < 6; ++step)
            {
                init();
                for (int k = 0; k < 3; ++k)
                {
                    adapt(mra_config().epsilon(1e-3));
                    init();
                }

                auto v = make_scalar_field<double>("v", mesh);
                v.fill(std::numeric_limits<double>::quiet_NaN());
                for_each_cell(mesh,
                              [&](auto& cell)
                              {
                                  v[cell] = value;
                              });
                if (outer_bc == OuterBc::neumann)
                {
                    make_bc<Neumann<1>>(v, 0.);
                }
                else if (outer_bc == OuterBc::dirichlet)
                {
                    make_bc<Dirichlet<1>>(v, value);
                }
                update_ghost_mr(v);

                auto check = [&](std::size_t level, auto&& set)
                {
                    set(
                        [&](const auto& i, const auto& index)
                        {
                            for (auto ii = i.start; ii < i.end; ++ii)
                            {
                                if (v(level, {ii, ii + 1}, index)[0] != value)
                                {
                                    ++n_wrong;
                                }
                            }
                        });
                };
                const auto& cells     = mesh[mesh_id_t::cells];
                const auto& reference = mesh[mesh_id_t::reference];
                for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
                {
                    check(level, intersection(nestedExpand(cells[level], mesh.ghost_width()), reference[level]));
                    check(level - 1,
                          intersection(nestedExpand(self(cells[level]).on(level - 1), prediction_stencil_radius), reference[level - 1]));
                    check(level - 2,
                          intersection(nestedExpand(self(cells[level]).on(level - 2), prediction_stencil_radius), reference[level - 2]));
                }
            }
            return n_wrong;
        }
    }

    // Control: refinement kept in the interior works.
    TEST(ghost_update, interior_refinement_3d)
    {
        EXPECT_NO_THROW(run_ghost_update_3d(/*boundary_touching=*/false));
    }

    // Regression: when the refined region reaches the domain boundary/corner in 3D,
    // update_ghost_mr projects the outer corner/edge ghosts two levels down
    // (project_corner_below). It reads the corner-most child at a reconstructed,
    // fixed-parity position; on such a mesh that child may not exist, so
    // LevelCellArray::get_interval indexed its storage with a negative offset cast to
    // size_t (out-of-bounds read, SIGSEGV). get_interval now raises std::out_of_range
    // in that case, and project_corner_below looks the child up per cell and copies it
    // only when it exists. The 2D analogue never triggered it.
    TEST(ghost_update, boundary_touching_refinement_3d)
    {
        EXPECT_NO_THROW(run_ghost_update_3d(/*boundary_touching=*/true));
    }

    // Regression: update_outer_ghosts(field) loops from max_level down to
    // min_level - 1, clamped at 0. The loop used to test level >= bound with a
    // size_t level, so with a min_level of 0 or 1 the bound is 0, the test is
    // always true and level wraps around to SIZE_MAX (out-of-bounds access).
    TEST(ghost_update, outer_ghosts_all_levels_down_to_level_zero)
    {
        constexpr std::size_t dim = 2;

        for (const std::size_t min_level : {0UL, 1UL, 2UL})
        {
            auto config = mesh_config<dim>().min_level(min_level).max_level(4);
            auto mesh   = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, config);

            auto u = make_scalar_field<double>("u", mesh, 0.);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              u[cell] = cell.center(0) + 2 * cell.center(1);
                          });
            make_bc<Dirichlet<1>>(u, 1.);
            auto expected = u;

            const std::size_t lowest_level = min_level > 0 ? min_level - 1 : 0;
            for (const std::size_t level : std::views::iota(lowest_level, mesh.max_level() + 1) | std::views::reverse)
            {
                update_outer_ghosts(level, expected);
            }
            update_outer_ghosts(u);

            EXPECT_EQ(u.array(), expected.array()) << "min_level = " << min_level;
        }
    }

    // The outer ghosts the solver reads are all filled, with a prediction of radius 1 and 2: in the corners
    // refined below their level, above the coarse cells along the walls, below min_level, and before the
    // field has any boundary condition (the initial adaptation of a solver).
    TEST(ghost_update, outer_ghosts_read_by_the_solver_are_filled)
    {
        for (const auto outer_bc : {OuterBc::neumann, OuterBc::dirichlet, OuterBc::none})
        {
            EXPECT_EQ(count_wrong_outer_ghosts<1>(outer_bc), 0) << "radius 1, B.C. " << static_cast<int>(outer_bc);
            EXPECT_EQ(count_wrong_outer_ghosts<2>(outer_bc), 0) << "radius 2, B.C. " << static_cast<int>(outer_bc);
        }
    }
}
