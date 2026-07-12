// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Characterization tests for the boundary conditions: they read the ghost
// values written by the bc/ machinery and compare them to an analytical
// oracle, freezing the current behavior before the bc/ refactoring.
//
// - Cartesian B.C. (Dirichlet, Neumann), uniform and adapted meshes: every
//   filled ghost equals f(ghost center), with f a polynomial the reconstruction
//   is exact on (degree = order for Dirichlet, degree 1 for Neumann).
// - Corner extrapolation: the diagonal ghosts reflect the field about the
//   domain corner, the off-diagonal ghosts copy the diagonal value.
// - Far ghost layers (ghost_width > 1): filled by polynomial extrapolation.
// - SetRegion: a B.C. restricted to a user subset via ->on(subset).

#include <cmath>

#include <gtest/gtest.h>

#include <samurai/bc.hpp>
#include <samurai/boundary.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    namespace
    {
        // Non-zero axis and its sign for a Cartesian direction.
        template <std::size_t dim>
        std::pair<std::size_t, int> axis_and_sign(const DirectionVector<dim>& direction)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (direction[d] != 0)
                {
                    return {d, direction[d]};
                }
            }
            return {0, 0};
        }

        // Uniform mesh (min_level == max_level) with a chosen ghost width.
        template <std::size_t dim>
        auto uniform_mesh(std::size_t level, int ghost_width)
        {
            auto cfg = mesh_config<dim>().min_level(level).max_level(level).max_stencil_size(2 * ghost_width);
            return mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, cfg);
        }

        // Adapted mesh whose boundary crosses several levels: a sharp indicator
        // ball centered on the origin corner refines the '-' boundaries near the
        // corner while the rest stays coarse.
        template <std::size_t dim>
        auto adapted_mesh(int ghost_width)
        {
            auto cfg  = mesh_config<dim>().min_level(2).max_level(5).max_stencil_size(2 * ghost_width);
            auto mesh = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, cfg);

            auto phi = make_scalar_field<double>("phi", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              auto c   = cell.center();
                              double r = 0.;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r += c[d] * c[d];
                              }
                              phi[cell] = (std::sqrt(r) < 0.3) ? 1. : 0.;
                          });
            make_bc<Dirichlet<1>>(phi, 0.);
            make_MRAdapt(phi)(mra_config().epsilon(1e-4));
            return mesh;
        }

        // Number of levels holding boundary cells in the given direction.
        template <class Mesh>
        std::size_t count_boundary_levels(const Mesh& mesh, const DirectionVector<Mesh::dim>& direction)
        {
            std::size_t n = 0;
            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                bool any = false;
                domain_boundary(mesh, level, direction)(
                    [&](const auto&, const auto&)
                    {
                        any = true;
                    });
                if (any)
                {
                    ++n;
                }
            }
            return n;
        }

        // Iterate over the `h` filled ghost layers on the +/-direction side, at
        // every level, and check that each ghost holds f(ghost center). Works on
        // both uniform and adapted meshes (domain_boundary gives the boundary
        // cells per level). Returns the number of ghosts checked.
        template <class Field, class F>
        std::size_t check_ghosts(Field& u, const DirectionVector<Field::dim>& direction, int h, F&& f)
        {
            using mesh_id_t = typename Field::mesh_t::mesh_id_t;
            auto& mesh      = u.mesh();

            std::size_t nb = 0;
            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                auto inner  = domain_boundary(mesh, level, direction);
                auto domain = self(mesh.domain()).on(level);
                for (int k = 1; k <= h; ++k)
                {
                    auto ghosts = intersection(difference(translate(inner, k * direction), domain), mesh[mesh_id_t::reference][level]).on(level);
                    for_each_cell(mesh[mesh_id_t::reference],
                                  ghosts,
                                  [&](auto& cell)
                                  {
                                      ++nb;
                                      EXPECT_NEAR(u[cell], f(cell.center()), 1e-11)
                                          << "dim=" << Field::dim << " direction=" << direction << " level=" << level << " layer=" << k;
                                  });
                }
            }
            return nb;
        }

        // Dirichlet of the given order. `functional` selects the FunctionBc path
        // (and adds a transverse-linear term, still exact for the normal-line
        // reconstruction) vs. the constant-value path.
        template <std::size_t dim, std::size_t order, class Mesh>
        void run_dirichlet(Mesh& mesh, bool functional)
        {
            const int h = static_cast<int>(order); // stencil_size / 2 = order ghost layers

            for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    auto [normal_axis, s] = axis_and_sign<dim>(direction);

                    // Polynomial 1 + 3x - 2x^2 in the normal coordinate (truncated to `order`),
                    // plus a transverse-linear term in the functional case.
                    auto f = [normal_axis, functional](const auto& coords)
                    {
                        double x = coords[normal_axis];
                        double p = 1. + 3. * x;
                        if constexpr (order >= 2)
                        {
                            p += -2. * x * x;
                        }
                        if (functional)
                        {
                            for (std::size_t d = 0; d < dim; ++d)
                            {
                                if (d != normal_axis)
                                {
                                    p += 0.5 * coords[d];
                                }
                            }
                        }
                        return p;
                    };

                    auto u = make_scalar_field<double>("u", mesh);
                    for_each_cell(mesh,
                                  [&](auto& cell)
                                  {
                                      u[cell] = f(cell.center());
                                  });

                    if (functional)
                    {
                        using cell_t   = typename decltype(u)::cell_t;
                        using coords_t = typename cell_t::coords_t;
                        make_bc<Dirichlet<order>>(u,
                                                  [f](const auto&, const cell_t&, const coords_t& coords)
                                                  {
                                                      return f(coords);
                                                  });
                    }
                    else
                    {
                        // Boundary value = f on the face (x = 0 on the '-' side, x = 1 on the '+' side).
                        xt::xtensor_fixed<double, xt::xshape<dim>> face;
                        face.fill(0.);
                        face[normal_axis] = (s > 0) ? 1. : 0.;
                        make_bc<Dirichlet<order>>(u, f(face));
                    }
                    apply_field_bc(u, DirectionVector<dim>(direction));

                    EXPECT_GT(check_ghosts(u, direction, h, f), 0u);
                });
        }

        // Neumann order 1: ghost = cell + dx * value with value the outward
        // normal derivative. Exact on linear f, so the ghost equals f(ghost
        // center). `functional` selects the FunctionBc path.
        template <std::size_t dim, class Mesh>
        void run_neumann(Mesh& mesh, bool functional)
        {
            const int h        = 1;
            const double slope = 3.;

            for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    auto [normal_axis, s] = axis_and_sign<dim>(direction);

                    auto f = [normal_axis, functional, slope](const auto& coords)
                    {
                        double p = 1. + slope * coords[normal_axis];
                        if (functional)
                        {
                            for (std::size_t d = 0; d < dim; ++d)
                            {
                                if (d != normal_axis)
                                {
                                    p += 0.5 * coords[d];
                                }
                            }
                        }
                        return p;
                    };

                    auto u = make_scalar_field<double>("u", mesh);
                    for_each_cell(mesh,
                                  [&](auto& cell)
                                  {
                                      u[cell] = f(cell.center());
                                  });

                    double normal_derivative = slope * s; // outward normal derivative of f

                    if (functional)
                    {
                        using cell_t   = typename decltype(u)::cell_t;
                        using coords_t = typename cell_t::coords_t;
                        make_bc<Neumann<1>>(u,
                                            [normal_derivative](const auto&, const cell_t&, const coords_t&)
                                            {
                                                return normal_derivative;
                                            });
                    }
                    else
                    {
                        make_bc<Neumann<1>>(u, normal_derivative);
                    }
                    apply_field_bc(u, DirectionVector<dim>(direction));

                    EXPECT_GT(check_ghosts(u, direction, h, f), 0u);
                });
        }

        // Number of axes along which the point lies outside the unit box.
        template <std::size_t dim>
        int count_outside(const xt::xtensor_fixed<double, xt::xshape<dim>>& c)
        {
            int nout = 0;
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (c[d] < 0. || c[d] > 1.)
                {
                    ++nout;
                }
            }
            return nout;
        }

        // Expected value of a corner-block ghost. update_outer_corners fills the
        // diagonal ghosts by an extrapolation that reflects the field about the
        // domain corner (exact up to degree 3), then copies each diagonal value
        // to the off-diagonal cells of the same first-axis layer. Both are
        // captured by: value == f(reflection of q), where q replaces every
        // outside-axis offset by the first outside-axis offset (keeping signs).
        template <std::size_t dim, class F>
        double corner_oracle(const xt::xtensor_fixed<double, xt::xshape<dim>>& c, F&& f)
        {
            std::array<int, dim> outward{};
            std::size_t first = 0;
            bool found        = false;
            for (std::size_t d = 0; d < dim; ++d)
            {
                outward[d] = (c[d] < 0.) ? -1 : (c[d] > 1.) ? 1 : 0;
                if (outward[d] != 0 && !found)
                {
                    first = d;
                    found = true;
                }
            }
            double r_first = outward[first] < 0 ? 0. : 1.;
            double mag     = std::abs(c[first] - r_first);

            xt::xtensor_fixed<double, xt::xshape<dim>> refl = c;
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (outward[d] != 0)
                {
                    double r = outward[d] < 0 ? 0. : 1.;
                    refl[d]  = r - mag * outward[d]; // reflect q = r + mag*outward about r
                }
            }
            return f(refl);
        }

        // Fill a uniform field with a polynomial, run the corner extrapolation
        // and check every corner-block ghost against corner_oracle.
        template <std::size_t dim>
        void run_corners(int ghost_width)
        {
            using mesh_id_t         = typename MRMesh<mesh_config<dim>>::mesh_id_t;
            const std::size_t level = 3;
            auto cfg = mesh_config<dim>().min_level(level).max_level(level).max_stencil_size(2 * ghost_width).disable_minimal_ghost_width();
            auto mesh = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, cfg);

            auto f = [](const auto& c)
            {
                double p = 1. + 0.5 * c[0] * c[0];
                for (std::size_t d = 0; d < dim; ++d)
                {
                    p += static_cast<double>(d + 1) * c[d];
                }
                return p;
            };

            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              u[cell] = f(cell.center());
                          });

            update_outer_corners_by_polynomial_extrapolation(level, u);

            std::size_t nb = 0;
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              auto c = cell.center();
                              if (count_outside<dim>(c) < 2)
                              {
                                  return; // face ghosts and inner cells are not corner ghosts
                              }
                              ++nb;
                              EXPECT_NEAR(u[cell], corner_oracle<dim>(c, f), 1e-11) << "dim=" << dim << " ghost_width=" << ghost_width;
                          });
            EXPECT_GT(nb, 0u);
        }

        // Fill the far ghost layers (beyond those written by the B.C.) with
        // update_further_ghosts_by_polynomial_extrapolation and check them. A
        // linear field with a first-order Dirichlet B.C. is exact for every
        // layer, so each ghost equals f(ghost center). ghost_width > 1 is
        // required for there to be far layers at all.
        template <std::size_t dim, class Mesh>
        void run_further(Mesh& mesh, int ghost_width)
        {
            auto f = [](const auto& c)
            {
                double p = 1.;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    p += static_cast<double>(d + 1) * c[d];
                }
                return p;
            };

            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              u[cell] = f(cell.center());
                          });

            using cell_t   = typename decltype(u)::cell_t;
            using coords_t = typename cell_t::coords_t;
            make_bc<Dirichlet<1>>(u,
                                  [f](const auto&, const cell_t&, const coords_t& c)
                                  {
                                      return f(c);
                                  });
            apply_field_bc(u);
            update_further_ghosts_by_polynomial_extrapolation(u);

            for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    EXPECT_GT(check_ghosts(u, DirectionVector<dim>(direction), ghost_width, f), 0u);
                });
        }
    }

    //-------------------------------------------------------------------------
    // Uniform mesh.
    //-------------------------------------------------------------------------
    TEST(bc_ghost_values, dirichlet1_constant_1d)
    {
        auto mesh = uniform_mesh<1>(4, 1);
        run_dirichlet<1, 1>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet1_constant_2d)
    {
        auto mesh = uniform_mesh<2>(4, 1);
        run_dirichlet<2, 1>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet1_constant_3d)
    {
        auto mesh = uniform_mesh<3>(4, 1);
        run_dirichlet<3, 1>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet1_function_1d)
    {
        auto mesh = uniform_mesh<1>(4, 1);
        run_dirichlet<1, 1>(mesh, true);
    }

    TEST(bc_ghost_values, dirichlet1_function_2d)
    {
        auto mesh = uniform_mesh<2>(4, 1);
        run_dirichlet<2, 1>(mesh, true);
    }

    TEST(bc_ghost_values, dirichlet1_function_3d)
    {
        auto mesh = uniform_mesh<3>(4, 1);
        run_dirichlet<3, 1>(mesh, true);
    }

    TEST(bc_ghost_values, dirichlet2_constant_1d)
    {
        auto mesh = uniform_mesh<1>(4, 2);
        run_dirichlet<1, 2>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet2_constant_2d)
    {
        auto mesh = uniform_mesh<2>(4, 2);
        run_dirichlet<2, 2>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet2_constant_3d)
    {
        auto mesh = uniform_mesh<3>(4, 2);
        run_dirichlet<3, 2>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet2_function_2d)
    {
        auto mesh = uniform_mesh<2>(4, 2);
        run_dirichlet<2, 2>(mesh, true);
    }

    TEST(bc_ghost_values, dirichlet2_function_3d)
    {
        auto mesh = uniform_mesh<3>(4, 2);
        run_dirichlet<3, 2>(mesh, true);
    }

    TEST(bc_ghost_values, neumann_constant_1d)
    {
        auto mesh = uniform_mesh<1>(4, 1);
        run_neumann<1>(mesh, false);
    }

    TEST(bc_ghost_values, neumann_constant_2d)
    {
        auto mesh = uniform_mesh<2>(4, 1);
        run_neumann<2>(mesh, false);
    }

    TEST(bc_ghost_values, neumann_constant_3d)
    {
        auto mesh = uniform_mesh<3>(4, 1);
        run_neumann<3>(mesh, false);
    }

    TEST(bc_ghost_values, neumann_function_2d)
    {
        auto mesh = uniform_mesh<2>(4, 1);
        run_neumann<2>(mesh, true);
    }

    TEST(bc_ghost_values, neumann_function_3d)
    {
        auto mesh = uniform_mesh<3>(4, 1);
        run_neumann<3>(mesh, true);
    }

    //-------------------------------------------------------------------------
    // Adapted mesh whose boundary crosses several levels.
    //-------------------------------------------------------------------------
    TEST(bc_ghost_values, adapted_boundary_crosses_levels_2d)
    {
        auto mesh                   = adapted_mesh<2>(1);
        DirectionVector<2> toward_0 = {-1, 0};
        EXPECT_GT(count_boundary_levels(mesh, toward_0), 1u);
    }

    TEST(bc_ghost_values, adapted_boundary_crosses_levels_3d)
    {
        auto mesh                   = adapted_mesh<3>(1);
        DirectionVector<3> toward_0 = {-1, 0, 0};
        EXPECT_GT(count_boundary_levels(mesh, toward_0), 1u);
    }

    TEST(bc_ghost_values, dirichlet1_adapted_2d)
    {
        auto mesh = adapted_mesh<2>(1);
        run_dirichlet<2, 1>(mesh, false);
        run_dirichlet<2, 1>(mesh, true);
    }

    TEST(bc_ghost_values, dirichlet1_adapted_3d)
    {
        auto mesh = adapted_mesh<3>(1);
        run_dirichlet<3, 1>(mesh, false);
        run_dirichlet<3, 1>(mesh, true);
    }

    TEST(bc_ghost_values, neumann_adapted_2d)
    {
        auto mesh = adapted_mesh<2>(1);
        run_neumann<2>(mesh, false);
        run_neumann<2>(mesh, true);
    }

    TEST(bc_ghost_values, neumann_adapted_3d)
    {
        auto mesh = adapted_mesh<3>(1);
        run_neumann<3>(mesh, false);
        run_neumann<3>(mesh, true);
    }

    TEST(bc_ghost_values, dirichlet2_adapted_2d)
    {
        auto mesh = adapted_mesh<2>(2);
        run_dirichlet<2, 2>(mesh, false);
    }

    //-------------------------------------------------------------------------
    // Corner (diagonal) ghosts by polynomial extrapolation.
    //-------------------------------------------------------------------------
    TEST(bc_ghost_values, corners_2d_ghost_width_1)
    {
        run_corners<2>(1);
    }

    TEST(bc_ghost_values, corners_2d_ghost_width_2)
    {
        run_corners<2>(2);
    }

    TEST(bc_ghost_values, corners_3d_ghost_width_1)
    {
        run_corners<3>(1);
    }

    TEST(bc_ghost_values, corners_3d_ghost_width_2)
    {
        run_corners<3>(2);
    }

    //-------------------------------------------------------------------------
    // Far ghost layers by polynomial extrapolation (ghost_width > 1).
    //-------------------------------------------------------------------------
    TEST(bc_ghost_values, further_ghosts_uniform_2d)
    {
        auto mesh = uniform_mesh<2>(4, 3);
        run_further<2>(mesh, 3);
    }

    TEST(bc_ghost_values, further_ghosts_uniform_3d)
    {
        auto mesh = uniform_mesh<3>(4, 3);
        run_further<3>(mesh, 3);
    }

    TEST(bc_ghost_values, further_ghosts_adapted_2d)
    {
        auto mesh = adapted_mesh<2>(3);
        run_further<2>(mesh, 3);
    }

    TEST(bc_ghost_values, further_ghosts_adapted_3d)
    {
        auto mesh = adapted_mesh<3>(3);
        run_further<3>(mesh, 3);
    }

    //-------------------------------------------------------------------------
    // SetRegion: restrict a boundary condition to a user-provided subset of the
    // boundary via make_bc(...)->on(subset).
    //-------------------------------------------------------------------------
    TEST(bc_ghost_values, set_region_restricts_to_subset)
    {
        static constexpr std::size_t dim = 2;
        const std::size_t level          = 3;
        auto cfg                         = mesh_config<dim>().min_level(level).max_level(level);
        auto mesh                        = mra::make_mesh(
            Box<double, dim>{
                {0., 0.},
                {1., 1.}
        },
            cfg);

        using mesh_t     = decltype(mesh);
        using interval_t = typename mesh_t::interval_t;
        using mesh_id_t  = typename mesh_t::mesh_id_t;
        using lcl_t      = LevelCellList<dim, interval_t>;
        using lca_t      = LevelCellArray<dim, interval_t>;

        // Left boundary column (i = 0), rows j = 2..5 only (corners at j = 0, 7 excluded).
        lcl_t lcl(level, mesh.origin_point(), mesh.scaling_factor());
        for (int j = 2; j < 6; ++j)
        {
            lcl[{j}].add_interval({0, 1});
        }
        lca_t sub(lcl);

        auto u = make_scalar_field<double>("u", mesh);
        u.fill(0.);
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          u[cell] = 5.;
                      });

        const double v = 1.;
        auto bc        = make_bc<Dirichlet<1>>(u, v)->on(self(sub));

        // The region holds a single direction (left) with exactly the 4 selected cells.
        const auto& region = bc->get_region();
        ASSERT_EQ(region.first.size(), 1u);
        EXPECT_EQ(region.first[0][0], -1);
        EXPECT_EQ(region.first[0][1], 0);
        EXPECT_EQ(region.second[0].nb_cells(), 4u);

        apply_field_bc(u);

        // Only the left ghosts at j = 2..5 are filled (ghost = 2v - cell); the
        // other left ghosts are left at their initial value.
        DirectionVector<dim> left = {-1, 0};
        auto inner                = domain_boundary(mesh, level, left);
        auto ghosts               = intersection(translate(inner, left), mesh[mesh_id_t::reference][level]).on(level);
        double dx                 = mesh.cell_length(level);
        std::size_t nb_filled     = 0;
        for_each_cell(mesh[mesh_id_t::reference],
                      ghosts,
                      [&](auto& cell)
                      {
                          int j = static_cast<int>(std::llround(cell.center(1) / dx - 0.5));
                          if (j >= 2 && j < 6)
                          {
                              EXPECT_NEAR(u[cell], 2. * v - 5., 1e-12) << "j=" << j;
                              ++nb_filled;
                          }
                          else
                          {
                              EXPECT_EQ(u[cell], 0.) << "j=" << j;
                          }
                      });
        EXPECT_EQ(nb_filled, 4u);
    }
}
