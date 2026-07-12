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
#include <map>
#include <string>

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

                    // Polynomial of degree `order` in the normal coordinate (the Dirichlet
                    // reconstruction of that order is exact on it), plus a transverse-linear
                    // term in the functional case.
                    auto f = [normal_axis, functional](const auto& coords)
                    {
                        double x = coords[normal_axis];
                        double p = 1. + 3. * x;
                        if constexpr (order >= 2)
                        {
                            p += -2. * x * x;
                        }
                        if constexpr (order >= 3)
                        {
                            p += 0.7 * x * x * x;
                        }
                        if constexpr (order >= 4)
                        {
                            p += -0.4 * x * x * x * x;
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

        // Expected value of a corner-block ghost. This oracle freezes the
        // *observed* behavior of the current implementation (it is not a claim
        // about the extrapolation order): on a uniform mesh update_outer_corners
        // fills the diagonal ghosts with the field reflected about the domain
        // corner, and copies each diagonal value to the off-diagonal cells of the
        // same first-axis layer. Both are captured by value == f(reflection of q),
        // where q replaces every outside-axis offset by the first outside-axis
        // offset (keeping signs). The reflection identity was verified for
        // polynomials up to degree 3; a change to the corner algorithm is
        // expected to require updating this oracle (see also the decoupled copy
        // invariant in corners_offdiagonal_copy_property).
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

        // Separable polynomial, degree `order` in each coordinate: along every
        // Cartesian normal its restriction has degree `order`, so a Dirichlet
        // B.C. of that order (and every higher-order polynomial extrapolation)
        // is exact on it.
        template <std::size_t dim, std::size_t order, class Coords>
        double separable_polynomial(const Coords& c)
        {
            double p = 1.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                double x = c[d];
                p += static_cast<double>(d + 1) * x;
                if constexpr (order >= 2)
                {
                    p += 0.3 * x * x;
                }
                if constexpr (order >= 3)
                {
                    p += 0.1 * x * x * x;
                }
            }
            return p;
        }

        // Fill the far ghost layers (beyond those written by the B.C.) with
        // update_further_ghosts_by_polynomial_extrapolation and check them.
        // A Dirichlet<order> (resp. Neumann<1>) B.C. on a degree-`order` (resp.
        // linear) separable field is exact for the near layers, and the far
        // layers are extrapolated exactly, so each ghost equals f(ghost center).
        // ghost_width must exceed the number of layers the B.C. fills (= order
        // for Dirichlet, 1 for Neumann) for there to be far layers at all.
        template <std::size_t dim, std::size_t order, bool is_neumann, class Mesh>
        void run_further(Mesh& mesh, int ghost_width)
        {
            auto f = [](const auto& c)
            {
                return separable_polynomial<dim, order>(c);
            };

            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              u[cell] = f(cell.center());
                          });

            using cell_t   = typename decltype(u)::cell_t;
            using coords_t = typename cell_t::coords_t;
            if constexpr (is_neumann)
            {
                // Outward normal derivative of the linear field: sum_d dir[d]*(d+1).
                make_bc<Neumann<1>>(u,
                                    [](const auto& dir, const cell_t&, const coords_t&)
                                    {
                                        double dd = 0.;
                                        for (std::size_t d = 0; d < dim; ++d)
                                        {
                                            dd += static_cast<double>(dir[d]) * static_cast<double>(d + 1);
                                        }
                                        return dd;
                                    });
            }
            else
            {
                make_bc<Dirichlet<order>>(u,
                                          [f](const auto&, const cell_t&, const coords_t& c)
                                          {
                                              return f(c);
                                          });
            }
            apply_field_bc(u);
            update_further_ghosts_by_polynomial_extrapolation(u);

            for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    EXPECT_GT(check_ghosts(u, DirectionVector<dim>(direction), ghost_width, f), 0u);
                });
        }

        // Vector-field variant of check_ghosts: fc(coords, comp) gives the
        // expected value of component `comp`.
        template <class Field, class FC>
        std::size_t check_vector_ghosts(Field& u, const DirectionVector<Field::dim>& direction, int h, FC&& fc)
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
                                      for (std::size_t comp = 0; comp < Field::n_comp; ++comp)
                                      {
                                          EXPECT_NEAR(u[cell](comp), fc(cell.center(), comp), 1e-11)
                                              << "direction=" << direction << " comp=" << comp;
                                      }
                                  });
                }
            }
            return nb;
        }

        // Vector Dirichlet<1> / Neumann<1> (functional path). Component `comp`
        // carries (comp + 1) times a scalar field the first-order reconstruction
        // is exact on, so each ghost component equals fc(ghost center, comp) and
        // the reconstruction is checked to act component by component.
        template <std::size_t dim, std::size_t n_comp, bool is_neumann, class Mesh>
        void run_vector(Mesh& mesh)
        {
            const double slope = 3.;

            for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    auto [normal_axis, s] = axis_and_sign<dim>(direction);

                    auto fc = [normal_axis, slope](const auto& coords, std::size_t comp)
                    {
                        return static_cast<double>(comp + 1) * (1. + slope * coords[normal_axis]);
                    };

                    auto u = make_vector_field<double, n_comp>("u", mesh);
                    for_each_cell(mesh,
                                  [&](auto& cell)
                                  {
                                      for (std::size_t comp = 0; comp < n_comp; ++comp)
                                      {
                                          u[cell](comp) = fc(cell.center(), comp);
                                      }
                                  });

                    using field_t  = decltype(u);
                    using cell_t   = typename field_t::cell_t;
                    using coords_t = typename cell_t::coords_t;
                    using value_t  = typename field_t::value_type;
                    if constexpr (is_neumann)
                    {
                        make_bc<Neumann<1>>(u,
                                            [slope, s](const auto&, const cell_t&, const coords_t&)
                                            {
                                                samurai::CollapsArray<value_t, n_comp, false> val;
                                                for (std::size_t comp = 0; comp < n_comp; ++comp)
                                                {
                                                    val(comp) = static_cast<double>(comp + 1) * slope * s;
                                                }
                                                return val;
                                            });
                    }
                    else
                    {
                        make_bc<Dirichlet<1>>(u,
                                              [fc](const auto&, const cell_t&, const coords_t& coords)
                                              {
                                                  samurai::CollapsArray<value_t, n_comp, false> val;
                                                  for (std::size_t comp = 0; comp < n_comp; ++comp)
                                                  {
                                                      val(comp) = fc(coords, comp);
                                                  }
                                                  return val;
                                              });
                    }
                    apply_field_bc(u, DirectionVector<dim>(direction));

                    EXPECT_GT(check_vector_ghosts(u, DirectionVector<dim>(direction), 1, fc), 0u);
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

    TEST(bc_ghost_values, dirichlet3_constant_1d)
    {
        auto mesh = uniform_mesh<1>(4, 3);
        run_dirichlet<1, 3>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet3_constant_2d)
    {
        auto mesh = uniform_mesh<2>(4, 3);
        run_dirichlet<2, 3>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet3_function_2d)
    {
        auto mesh = uniform_mesh<2>(4, 3);
        run_dirichlet<2, 3>(mesh, true);
    }

    TEST(bc_ghost_values, dirichlet4_constant_1d)
    {
        auto mesh = uniform_mesh<1>(4, 4);
        run_dirichlet<1, 4>(mesh, false);
    }

    TEST(bc_ghost_values, dirichlet4_constant_2d)
    {
        auto mesh = uniform_mesh<2>(4, 4);
        run_dirichlet<2, 4>(mesh, false);
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
    // Vector fields (reconstruction applied component by component).
    //-------------------------------------------------------------------------
    TEST(bc_ghost_values, vector_dirichlet_2d)
    {
        auto mesh = uniform_mesh<2>(4, 1);
        run_vector<2, 3, false>(mesh);
    }

    TEST(bc_ghost_values, vector_dirichlet_3d)
    {
        auto mesh = uniform_mesh<3>(4, 1);
        run_vector<3, 2, false>(mesh);
    }

    TEST(bc_ghost_values, vector_neumann_2d)
    {
        auto mesh = uniform_mesh<2>(4, 1);
        run_vector<2, 3, true>(mesh);
    }

    TEST(bc_ghost_values, vector_neumann_3d)
    {
        auto mesh = uniform_mesh<3>(4, 1);
        run_vector<3, 2, true>(mesh);
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

    // Copy invariant, independent of the reflection oracle: within a corner
    // block, all ghosts that share the same outward-sign pattern and the same
    // offset along the first outside axis hold the same value (the off-diagonal
    // ghosts copy the diagonal value of their layer). Uses an asymmetric field
    // so the check would fail if the copy structure were wrong.
    TEST(bc_ghost_values, corners_offdiagonal_copy_property)
    {
        static constexpr std::size_t dim = 2;
        using mesh_id_t                  = typename MRMesh<mesh_config<dim>>::mesh_id_t;
        const std::size_t level          = 3;
        auto cfg  = mesh_config<dim>().min_level(level).max_level(level).max_stencil_size(4).disable_minimal_ghost_width();
        auto mesh = mra::make_mesh(
            Box<double, dim>{
                {0., 0.},
                {1., 1.}
        },
            cfg);

        auto u = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          auto c  = cell.center();
                          u[cell] = 1. + 2. * c[0] + 5. * c[1] + 3. * c[0] * c[1];
                      });
        update_outer_corners_by_polynomial_extrapolation(level, u);

        double dx = mesh.cell_length(level);
        std::map<std::string, double> by_layer;
        std::size_t nb = 0;
        for_each_cell(mesh[mesh_id_t::reference],
                      [&](auto& cell)
                      {
                          auto c = cell.center();
                          if (count_outside<dim>(c) < 2)
                          {
                              return;
                          }
                          std::string key;
                          std::size_t first = dim;
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              int s = (c[d] < 0.) ? -1 : (c[d] > 1.) ? 1 : 0;
                              key += std::to_string(s) + ",";
                              if (s != 0 && first == dim)
                              {
                                  first = d;
                              }
                          }
                          double r_first = c[first] < 0. ? 0. : 1.;
                          int layer      = static_cast<int>(std::llround(std::abs(c[first] - r_first) / dx + 0.5));
                          key += "|" + std::to_string(layer);

                          auto it = by_layer.find(key);
                          if (it == by_layer.end())
                          {
                              by_layer[key] = u[cell];
                          }
                          else
                          {
                              EXPECT_NEAR(u[cell], it->second, 1e-12) << "key=" << key;
                          }
                          ++nb;
                      });
        EXPECT_GT(nb, 0u);
    }

    //-------------------------------------------------------------------------
    // Far ghost layers by polynomial extrapolation (ghost_width > 1).
    //-------------------------------------------------------------------------
    TEST(bc_ghost_values, further_ghosts_dirichlet1_uniform_2d)
    {
        auto mesh = uniform_mesh<2>(4, 3);
        run_further<2, 1, false>(mesh, 3);
    }

    TEST(bc_ghost_values, further_ghosts_dirichlet1_uniform_3d)
    {
        auto mesh = uniform_mesh<3>(4, 3);
        run_further<3, 1, false>(mesh, 3);
    }

    TEST(bc_ghost_values, further_ghosts_dirichlet1_adapted_2d)
    {
        auto mesh = adapted_mesh<2>(3);
        run_further<2, 1, false>(mesh, 3);
    }

    TEST(bc_ghost_values, further_ghosts_dirichlet1_adapted_3d)
    {
        auto mesh = adapted_mesh<3>(3);
        run_further<3, 1, false>(mesh, 3);
    }

    // Dirichlet<2> fills 2 near layers; layer 3 is the one extrapolated (stencil
    // size 6, the largest the polynomial extrapolation implements).
    TEST(bc_ghost_values, further_ghosts_dirichlet2_uniform_2d)
    {
        auto mesh = uniform_mesh<2>(4, 3);
        run_further<2, 2, false>(mesh, 3);
    }

    TEST(bc_ghost_values, further_ghosts_neumann_uniform_2d)
    {
        auto mesh = uniform_mesh<2>(4, 3);
        run_further<2, 1, true>(mesh, 3);
    }

    TEST(bc_ghost_values, further_ghosts_neumann_uniform_3d)
    {
        auto mesh = uniform_mesh<3>(4, 3);
        run_further<3, 1, true>(mesh, 3);
    }

    //-------------------------------------------------------------------------
    // SetRegion: restrict a boundary condition to a user-provided subset of the
    // boundary via make_bc(...)->on(subset).
    //-------------------------------------------------------------------------
    namespace
    {
        // LCA of one 2D boundary strip: the cells adjacent to side `side` (-1/+1)
        // of axis `axis`, restricted to the transverse index range [tlo, thi).
        template <class Mesh>
        auto boundary_strip(const Mesh& mesh, std::size_t level, std::size_t axis, int side, int tlo, int thi)
        {
            using interval_t = typename Mesh::interval_t;
            using lcl_t      = LevelCellList<2, interval_t>;
            using lca_t      = LevelCellArray<2, interval_t>;

            int n = 1 << level;
            int b = (side < 0) ? 0 : n - 1; // boundary index along `axis`

            lcl_t lcl(level, mesh.origin_point(), mesh.scaling_factor());
            if (axis == 0)
            {
                for (int t = tlo; t < thi; ++t)
                {
                    lcl[{t}].add_interval({b, b + 1});
                }
            }
            else
            {
                lcl[{b}].add_interval({tlo, thi});
            }
            return lca_t(lcl);
        }

        auto unit_square_mesh(std::size_t level, int ghost_width)
        {
            auto cfg = mesh_config<2>().min_level(level).max_level(level).max_stencil_size(2 * ghost_width);
            return mra::make_mesh(
                Box<double, 2>{
                    {0., 0.},
                    {1., 1.}
            },
                cfg);
        }
    }

    TEST(bc_ghost_values, set_region_restricts_to_subset)
    {
        const std::size_t level = 3;
        auto mesh               = unit_square_mesh(level, 1);
        using mesh_id_t         = typename decltype(mesh)::mesh_id_t;

        // Left boundary column, rows j = 2..5 only (corners at j = 0, 7 excluded).
        auto sub = boundary_strip(mesh, level, 0, -1, 2, 6);

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
        DirectionVector<2> left = {-1, 0};
        auto inner              = domain_boundary(mesh, level, left);
        auto ghosts             = intersection(translate(inner, left), mesh[mesh_id_t::reference][level]).on(level);
        std::size_t nb_filled   = 0;
        for_each_cell(mesh[mesh_id_t::reference],
                      ghosts,
                      [&](auto& cell)
                      {
                          int j = static_cast<int>(cell.indices[1]);
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

    // The region direction has the correct axis and sign for each of the four sides.
    TEST(bc_ghost_values, set_region_direction_per_side)
    {
        const std::size_t level = 3;
        auto mesh               = unit_square_mesh(level, 1);

        for (std::size_t axis = 0; axis < 2; ++axis)
        {
            for (int side : {-1, 1})
            {
                auto sub = boundary_strip(mesh, level, axis, side, 2, 6); // mid-edge, no corners
                auto u   = make_scalar_field<double>("u", mesh);
                u.fill(0.);
                auto bc = make_bc<Dirichlet<1>>(u, 1.)->on(self(sub));

                const auto& region = bc->get_region();
                ASSERT_EQ(region.first.size(), 1u) << "axis=" << axis << " side=" << side;
                EXPECT_EQ(region.first[0][axis], side);
                EXPECT_EQ(region.first[0][1 - axis], 0);
                EXPECT_EQ(region.second[0].nb_cells(), 4u);
            }
        }
    }

    // A subset spanning the whole left column also touches the two corners, so the
    // region holds several directions.
    TEST(bc_ghost_values, set_region_multiple_directions)
    {
        const std::size_t level = 3;
        auto mesh               = unit_square_mesh(level, 1);

        auto sub = boundary_strip(mesh, level, 0, -1, 0, 1 << level); // full left column
        auto u   = make_scalar_field<double>("u", mesh);
        u.fill(0.);
        auto bc = make_bc<Dirichlet<1>>(u, 1.)->on(self(sub));

        const auto& region = bc->get_region();
        EXPECT_GT(region.first.size(), 1u);
        bool has_left = false;
        for (const auto& d : region.first)
        {
            if (d[0] == -1 && d[1] == 0)
            {
                has_left = true;
            }
        }
        EXPECT_TRUE(has_left);
    }

    // SetRegion works with a Neumann condition.
    TEST(bc_ghost_values, set_region_neumann)
    {
        const std::size_t level = 3;
        auto mesh               = unit_square_mesh(level, 1);
        using mesh_id_t         = typename decltype(mesh)::mesh_id_t;

        auto sub = boundary_strip(mesh, level, 0, -1, 2, 6);
        auto u   = make_scalar_field<double>("u", mesh);
        u.fill(0.);
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          u[cell] = 5.;
                      });

        const double v = 2.;
        auto bc        = make_bc<Neumann<1>>(u, v)->on(self(sub));

        const auto& region = bc->get_region();
        ASSERT_EQ(region.first.size(), 1u);
        EXPECT_EQ(region.first[0][0], -1);

        apply_field_bc(u);

        double dx               = mesh.cell_length(level);
        DirectionVector<2> left = {-1, 0};
        auto inner              = domain_boundary(mesh, level, left);
        auto ghosts             = intersection(translate(inner, left), mesh[mesh_id_t::reference][level]).on(level);
        std::size_t nb_filled   = 0;
        for_each_cell(mesh[mesh_id_t::reference],
                      ghosts,
                      [&](auto& cell)
                      {
                          int j = static_cast<int>(cell.indices[1]);
                          if (j >= 2 && j < 6)
                          {
                              EXPECT_NEAR(u[cell], 5. + dx * v, 1e-12) << "j=" << j;
                              ++nb_filled;
                          }
                          else
                          {
                              EXPECT_EQ(u[cell], 0.) << "j=" << j;
                          }
                      });
        EXPECT_EQ(nb_filled, 4u);
    }

    // SetRegion works with a two-layer (order 2) condition: both ghost layers of
    // the selected cells are filled, the others are left untouched.
    TEST(bc_ghost_values, set_region_dirichlet2)
    {
        const std::size_t level = 3;
        auto mesh               = unit_square_mesh(level, 2);
        using mesh_id_t         = typename decltype(mesh)::mesh_id_t;

        auto sub = boundary_strip(mesh, level, 0, -1, 2, 6);
        auto u   = make_scalar_field<double>("u", mesh);
        u.fill(0.);
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          u[cell] = 5.;
                      });

        auto bc = make_bc<Dirichlet<2>>(u, 1.)->on(self(sub));

        const auto& region = bc->get_region();
        ASSERT_EQ(region.first.size(), 1u);
        EXPECT_EQ(region.first[0][0], -1);

        apply_field_bc(u);

        DirectionVector<2> left = {-1, 0};
        auto inner              = domain_boundary(mesh, level, left);
        for (int layer = 1; layer <= 2; ++layer)
        {
            auto ghosts           = intersection(translate(inner, layer * left), mesh[mesh_id_t::reference][level]).on(level);
            std::size_t nb_filled = 0;
            for_each_cell(mesh[mesh_id_t::reference],
                          ghosts,
                          [&](auto& cell)
                          {
                              int j = static_cast<int>(cell.indices[1]);
                              if (j >= 2 && j < 6)
                              {
                                  EXPECT_NE(u[cell], 0.) << "layer=" << layer << " j=" << j; // filled
                                  ++nb_filled;
                              }
                              else
                              {
                                  EXPECT_EQ(u[cell], 0.) << "layer=" << layer << " j=" << j; // untouched
                              }
                          });
            EXPECT_EQ(nb_filled, 4u);
        }
    }

    // SetRegion on an adapted mesh: restricting to the whole left boundary yields a
    // non-empty region containing the left direction.
    TEST(bc_ghost_values, set_region_adapted)
    {
        auto mesh = adapted_mesh<2>(1);

        auto left_boundary = difference(mesh.domain(), translate(mesh.domain(), DirectionVector<2>{1, 0}));
        auto u             = make_scalar_field<double>("u", mesh);
        u.fill(0.);
        auto bc = make_bc<Dirichlet<1>>(u, 1.)->on(left_boundary);

        const auto& region = bc->get_region();
        ASSERT_GT(region.first.size(), 0u);
        bool has_left = false;
        for (const auto& d : region.first)
        {
            if (d[0] == -1 && d[1] == 0)
            {
                has_left = true;
            }
        }
        EXPECT_TRUE(has_left);
    }

    //-------------------------------------------------------------------------
    // Mixed conditions: Dirichlet on one side, Neumann on the opposite side.
    //-------------------------------------------------------------------------
    TEST(bc_ghost_values, mixed_dirichlet_neumann)
    {
        const std::size_t level = 3;
        auto mesh               = unit_square_mesh(level, 1);
        using mesh_id_t         = typename decltype(mesh)::mesh_id_t;

        auto u = make_scalar_field<double>("u", mesh);
        u.fill(0.);
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          u[cell] = 5.;
                      });

        const double vD = 1.;
        const double vN = 2.;
        make_bc<Dirichlet<1>>(u, vD)->on(DirectionVector<2>{-1, 0}); // left
        make_bc<Neumann<1>>(u, vN)->on(DirectionVector<2>{1, 0});    // right

        apply_field_bc(u);

        double dx = mesh.cell_length(level);
        for (int side : {-1, 1})
        {
            DirectionVector<2> dir = {side, 0};
            auto inner             = domain_boundary(mesh, level, dir);
            auto ghosts            = intersection(translate(inner, dir), mesh[mesh_id_t::reference][level]).on(level);
            for_each_cell(mesh[mesh_id_t::reference],
                          ghosts,
                          [&](auto& cell)
                          {
                              if (side < 0)
                              {
                                  EXPECT_NEAR(u[cell], 2. * vD - 5., 1e-12); // Dirichlet
                              }
                              else
                              {
                                  EXPECT_NEAR(u[cell], 5. + dx * vN, 1e-12); // Neumann
                              }
                          });
        }
    }
}
