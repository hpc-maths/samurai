#include <cmath>
#include <ranges>

#include <gtest/gtest.h>

#include <samurai/algorithm/update_outer_ghost.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

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
    }

    // Control: refinement kept in the interior works.
    TEST(ghost_update, interior_refinement_3d)
    {
        ::samurai::initialize();
        EXPECT_NO_THROW(run_ghost_update_3d(/*boundary_touching=*/false));
        ::samurai::finalize();
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
        ::samurai::initialize();
        EXPECT_NO_THROW(run_ghost_update_3d(/*boundary_touching=*/true));
        ::samurai::finalize();
    }

    // Regression: update_outer_ghosts(field) loops from max_level down to
    // min_level - 1, clamped at 0. The loop used to test level >= bound with a
    // size_t level, so with a min_level of 0 or 1 the bound is 0, the test is
    // always true and level wraps around to SIZE_MAX (out-of-bounds access).
    TEST(ghost_update, outer_ghosts_all_levels_down_to_level_zero)
    {
        ::samurai::initialize();

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

        ::samurai::finalize();
    }
}
