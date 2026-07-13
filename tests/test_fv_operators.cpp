// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Characterization tests for the finite-volume operators of schemes/fv/.
//
// Before any refactoring of schemes/fv/ (Improvement guide 04), these tests
// freeze the current behaviour of the explicit operators on:
//   - a tiny uniform mesh   -> local exactness on polynomials (interior cells);
//   - a refined uniform mesh -> convergence order;
//   - a hand-built two-level mesh -> conservation and level-jump fluxes;
//   - (with PETSc) explicit application vs assembled matrix.
//
// The assertions target *interior* cells (far enough from the boundary that the
// stencil only reads real cells), so they do not depend on the boundary
// condition. A boundary condition is nonetheless attached to every field
// because the explicit application applies it while filling the ghosts.

#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include <samurai/algorithm/update_ghost_mr.hpp>
#include <samurai/bc.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

#ifdef SAMURAI_WITH_PETSC
#include <samurai/petsc/utils.hpp>
#endif

namespace samurai
{
    namespace
    {
        // ---------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------

        // Uniform mesh on the unit box [0,1]^dim with 2^level cells per direction.
        // `max_stencil_size` must be raised for wide schemes such as WENO5 (6).
        template <std::size_t dim>
        auto uniform_mesh(std::size_t level, std::size_t max_stencil_size = 2)
        {
            using Box     = samurai::Box<double, dim>;
            using point_t = typename Box::point_t;
            point_t corner1, corner2;
            corner1.fill(0.);
            corner2.fill(1.);
            Box box(corner1, corner2);
            auto cfg = samurai::mesh_config<dim>().min_level(level).max_level(level).max_stencil_size(static_cast<int>(max_stencil_size));
            return samurai::mra::make_mesh(box, cfg);
        }

        // A cell is "interior" for a stencil of half-width `half_width` when every
        // index stays at distance >= half_width from both domain borders. On such a
        // cell the operator stencil only reads real cells, so the result does not
        // depend on the boundary condition.
        template <class Cell>
        bool is_interior(const Cell& cell, std::size_t nb_cells_per_dir, std::size_t half_width)
        {
            const auto w  = static_cast<long long>(half_width);
            const auto nx = static_cast<long long>(nb_cells_per_dir);
            for (std::size_t d = 0; d < Cell::dim; ++d)
            {
                const auto i = static_cast<long long>(cell.indices[d]);
                if (i < w || i > nx - 1 - w)
                {
                    return false;
                }
            }
            return true;
        }

        constexpr double tol = 1e-11;
    }

    // =====================================================================
    //  Local exactness on a uniform mesh
    // =====================================================================

    // Diffusion discretizes -Laplacian (with a diagonal diffusion tensor K).
    // For u = sum_d ( K-independent quadratic ), -div(K grad u) is the constant
    //   -sum_d K_d * d^2u/dx_d^2.
    template <std::size_t dim>
    void check_diffusion_exact()
    {
        const std::size_t level = 3; // 8 cells per direction
        const std::size_t nx    = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_scalar_field<double>("u", mesh);

        // u(x) = sum_d ( (d+1) x_d^2 + (d+2) x_d ) + 3
        auto exact = [](const auto& x)
        {
            double v = 3.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                v += static_cast<double>(d + 1) * x(d) * x(d) + static_cast<double>(d + 2) * x(d);
            }
            return v;
        };
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = exact(cell.center());
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        DiffCoeff<dim> K;
        for (std::size_t d = 0; d < dim; ++d)
        {
            K(d) = static_cast<double>(d + 1);
        }
        // -sum_d K_d * (2*(d+1))
        double expected = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            expected -= K(d) * 2. * static_cast<double>(d + 1);
        }

        auto diff   = make_diffusion_order2<decltype(u)>(K);
        auto result = diff(u);

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              EXPECT_NEAR(result[cell], expected, tol) << "dim=" << dim;
                              ++nb_checked;
                          }
                      });
        EXPECT_GT(nb_checked, 0u);
    }

    TEST(fv_operators, diffusion_exact_1d)
    {
        check_diffusion_exact<1>();
    }

    TEST(fv_operators, diffusion_exact_2d)
    {
        check_diffusion_exact<2>();
    }

    TEST(fv_operators, diffusion_exact_3d)
    {
        check_diffusion_exact<3>();
    }

    // Linear (upwind) convection with constant velocity computes div(v u) = v.grad(u).
    // For a linear field u = sum_d b_d x_d, this is the constant sum_d v_d b_d,
    // reproduced exactly by upwind (backward/forward difference of a linear field).
    template <std::size_t dim>
    void check_convection_exact()
    {
        const std::size_t level = 3;
        const std::size_t nx    = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_scalar_field<double>("u", mesh);

        std::array<double, dim> b;
        for (std::size_t d = 0; d < dim; ++d)
        {
            b[d] = static_cast<double>(d + 2);
        }
        auto exact = [&](const auto& x)
        {
            double v = 0.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                v += b[d] * x(d);
            }
            return v;
        };
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = exact(cell.center());
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        VelocityVector<dim> velocity;
        double expected = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            velocity(d) = static_cast<double>(d + 1);
            expected += velocity(d) * b[d];
        }

        auto conv   = make_convection_upwind<decltype(u)>(velocity);
        auto result = conv(u);

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              EXPECT_NEAR(result[cell], expected, tol) << "dim=" << dim;
                              ++nb_checked;
                          }
                      });
        EXPECT_GT(nb_checked, 0u);
    }

    TEST(fv_operators, convection_upwind_exact_1d)
    {
        check_convection_exact<1>();
    }

    TEST(fv_operators, convection_upwind_exact_2d)
    {
        check_convection_exact<2>();
    }

    TEST(fv_operators, convection_upwind_exact_3d)
    {
        check_convection_exact<3>();
    }

    // Gradient of a linear scalar field u = sum_d b_d x_d + c is the constant vector b.
    template <std::size_t dim>
    void check_gradient_exact()
    {
        const std::size_t level = 3;
        const std::size_t nx    = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_scalar_field<double>("u", mesh);

        std::array<double, dim> b;
        for (std::size_t d = 0; d < dim; ++d)
        {
            b[d] = static_cast<double>(d + 1);
        }
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          double v = 0.5;
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              v += b[d] * cell.center(d);
                          }
                          u[cell] = v;
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        auto grad   = make_gradient_order2<decltype(u)>();
        auto result = grad(u);

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  EXPECT_NEAR(result[cell](d), b[d], tol) << "dim=" << dim << " d=" << d;
                              }
                              ++nb_checked;
                          }
                      });
        EXPECT_GT(nb_checked, 0u);
    }

    TEST(fv_operators, gradient_exact_1d)
    {
        check_gradient_exact<1>();
    }

    TEST(fv_operators, gradient_exact_2d)
    {
        check_gradient_exact<2>();
    }

    TEST(fv_operators, gradient_exact_3d)
    {
        check_gradient_exact<3>();
    }

    // Divergence of a linear vector field u_d = sum_j A_dj x_j + c_d is the
    // constant sum_d A_dd.
    template <std::size_t dim>
    void check_divergence_exact()
    {
        const std::size_t level = 3;
        const std::size_t nx    = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_vector_field<double, dim>("u", mesh);

        // A_dj = (d+1) + j ; diagonal terms A_dd used for the expected divergence.
        auto A = [](std::size_t d, std::size_t j)
        {
            return static_cast<double>(d + 1 + j);
        };
        double expected = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            expected += A(d, d);
        }
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              double v = static_cast<double>(d);
                              for (std::size_t j = 0; j < dim; ++j)
                              {
                                  v += A(d, j) * cell.center(j);
                              }
                              u[cell](d) = v;
                          }
                      });
        // Homogeneous Dirichlet on every component (functional form works for any n_comp).
        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> zero;
                                  zero.fill(0.);
                                  return zero;
                              });

        auto div    = make_divergence_order2<decltype(u)>();
        auto result = div(u);

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              EXPECT_NEAR(result[cell], expected, tol) << "dim=" << dim;
                              ++nb_checked;
                          }
                      });
        EXPECT_GT(nb_checked, 0u);
    }

    TEST(fv_operators, divergence_exact_1d)
    {
        check_divergence_exact<1>();
    }

    TEST(fv_operators, divergence_exact_2d)
    {
        check_divergence_exact<2>();
    }

    TEST(fv_operators, divergence_exact_3d)
    {
        check_divergence_exact<3>();
    }

    // Same operators applied to a vector field (n_comp == dim), to cover the
    // vector code path of the schemes (n_comp x n_comp coefficient matrices).
    // Each component is an independent polynomial, so the operator acts
    // component-wise. Includes 1D (n_comp == 1), i.e. the VectorField<1> path.
    template <std::size_t dim>
    void check_diffusion_exact_vector()
    {
        const std::size_t level = 3;
        const std::size_t nx    = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_vector_field<double, dim>("u", mesh);

        // component c: u_c(x) = sum_d (c+1)(d+1) x_d^2
        auto q = [](std::size_t c, std::size_t d)
        {
            return static_cast<double>((c + 1) * (d + 1));
        };
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          for (std::size_t c = 0; c < dim; ++c)
                          {
                              double v = 0.;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  v += q(c, d) * cell.center(d) * cell.center(d);
                              }
                              u[cell](c) = v;
                          }
                      });
        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> zero;
                                  zero.fill(0.);
                                  return zero;
                              });

        DiffCoeff<dim> K;
        for (std::size_t d = 0; d < dim; ++d)
        {
            K(d) = static_cast<double>(d + 1);
        }
        // result_c = -sum_d K_d * 2 q(c,d)
        std::array<double, dim> expected;
        for (std::size_t c = 0; c < dim; ++c)
        {
            double e = 0.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                e -= K(d) * 2. * q(c, d);
            }
            expected[c] = e;
        }

        auto diff   = make_diffusion_order2<decltype(u)>(K);
        auto result = diff(u);

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              for (std::size_t c = 0; c < dim; ++c)
                              {
                                  EXPECT_NEAR(result[cell](c), expected[c], tol) << "dim=" << dim << " c=" << c;
                              }
                              ++nb_checked;
                          }
                      });
        EXPECT_GT(nb_checked, 0u);
    }

    TEST(fv_operators, diffusion_vector_exact_1d)
    {
        check_diffusion_exact_vector<1>();
    }

    TEST(fv_operators, diffusion_vector_exact_2d)
    {
        check_diffusion_exact_vector<2>();
    }

    TEST(fv_operators, diffusion_vector_exact_3d)
    {
        check_diffusion_exact_vector<3>();
    }

    // Vector convection: div(v u_c) = v.grad(u_c) component-wise, exact on linear
    // components.
    template <std::size_t dim>
    void check_convection_exact_vector()
    {
        const std::size_t level = 3;
        const std::size_t nx    = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_vector_field<double, dim>("u", mesh);

        auto b = [](std::size_t c, std::size_t d)
        {
            return static_cast<double>(c + d + 1);
        };
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          for (std::size_t c = 0; c < dim; ++c)
                          {
                              double v = 0.;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  v += b(c, d) * cell.center(d);
                              }
                              u[cell](c) = v;
                          }
                      });
        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> zero;
                                  zero.fill(0.);
                                  return zero;
                              });

        VelocityVector<dim> velocity;
        for (std::size_t d = 0; d < dim; ++d)
        {
            velocity(d) = static_cast<double>(d + 1);
        }

        // Component-wise advection: result_c = v . grad(u_c) = sum_d v_d du_c/dx_d,
        // with du_c/dx_d = b(c, d).
        std::array<double, dim> expected;
        for (std::size_t c = 0; c < dim; ++c)
        {
            double e = 0.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                e += velocity(d) * b(c, d);
            }
            expected[c] = e;
        }

        auto conv   = make_convection_upwind<decltype(u)>(velocity);
        auto result = conv(u);

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              for (std::size_t c = 0; c < dim; ++c)
                              {
                                  EXPECT_NEAR(result[cell](c), expected[c], tol) << "dim=" << dim << " c=" << c;
                              }
                              ++nb_checked;
                          }
                      });
        EXPECT_GT(nb_checked, 0u);
    }

    TEST(fv_operators, convection_vector_exact_1d)
    {
        check_convection_exact_vector<1>();
    }

    TEST(fv_operators, convection_vector_exact_2d)
    {
        check_convection_exact_vector<2>();
    }

    // Same as above, but through the *variable-velocity* overload
    // make_convection_upwind(VelocityField&) (the LinearHeterogeneous path used in
    // production, e.g. lid_driven_cavity). The velocity is a constant field, so the
    // face-averaged velocity equals that constant and the expected result matches
    // the constant-velocity case. Exercises the (fixed) diagonal branch of the
    // heterogeneous flux for a genuine vector field.
    template <std::size_t dim>
    void check_convection_variable_velocity_vector()
    {
        const std::size_t level = 3;
        const std::size_t nx    = 1u << level;

        auto mesh = uniform_mesh<dim>(level);

        auto w = [](std::size_t d)
        {
            return static_cast<double>(d + 1);
        };
        auto velocity_field = make_vector_field<double, dim>("velocity", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              velocity_field[cell](d) = w(d);
                          }
                      });
        make_bc<Dirichlet<1>>(velocity_field,
                              [&](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> v;
                                  for (std::size_t d = 0; d < dim; ++d)
                                  {
                                      v(d) = w(d);
                                  }
                                  return v;
                              });

        auto b = [](std::size_t c, std::size_t d)
        {
            return static_cast<double>(c + d + 1);
        };
        auto u = make_vector_field<double, dim>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          for (std::size_t c = 0; c < dim; ++c)
                          {
                              double v = 0.;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  v += b(c, d) * cell.center(d);
                              }
                              u[cell](c) = v;
                          }
                      });
        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> zero;
                                  zero.fill(0.);
                                  return zero;
                              });

        std::array<double, dim> expected;
        for (std::size_t c = 0; c < dim; ++c)
        {
            double e = 0.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                e += w(d) * b(c, d);
            }
            expected[c] = e;
        }

        auto conv   = make_convection_upwind<decltype(u)>(velocity_field);
        auto result = conv(u);

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              for (std::size_t c = 0; c < dim; ++c)
                              {
                                  EXPECT_NEAR(result[cell](c), expected[c], tol) << "dim=" << dim << " c=" << c;
                              }
                              ++nb_checked;
                          }
                      });
        EXPECT_GT(nb_checked, 0u);
    }

    TEST(fv_operators, convection_variable_velocity_vector_2d)
    {
        check_convection_variable_velocity_vector<2>();
    }

    TEST(fv_operators, convection_variable_velocity_vector_3d)
    {
        check_convection_variable_velocity_vector<3>();
    }

    // Variable-velocity convection of a scalar field (the n_comp == 1 branch of the
    // heterogeneous flux).
    template <std::size_t dim>
    void check_convection_variable_velocity_scalar()
    {
        const std::size_t level = 3;
        const std::size_t nx    = 1u << level;

        auto mesh = uniform_mesh<dim>(level);

        auto w = [](std::size_t d)
        {
            return static_cast<double>(d + 1);
        };
        auto velocity_field = make_vector_field<double, dim>("velocity", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              velocity_field[cell](d) = w(d);
                          }
                      });
        make_bc<Dirichlet<1>>(velocity_field,
                              [&](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> v;
                                  for (std::size_t d = 0; d < dim; ++d)
                                  {
                                      v(d) = w(d);
                                  }
                                  return v;
                              });

        std::array<double, dim> b;
        double expected = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            b[d] = static_cast<double>(d + 2);
            expected += w(d) * b[d];
        }
        auto u = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          double v = 0.;
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              v += b[d] * cell.center(d);
                          }
                          u[cell] = v;
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        auto conv   = make_convection_upwind<decltype(u)>(velocity_field);
        auto result = conv(u);

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              EXPECT_NEAR(result[cell], expected, tol) << "dim=" << dim;
                              ++nb_checked;
                          }
                      });
        EXPECT_GT(nb_checked, 0u);
    }

    TEST(fv_operators, convection_variable_velocity_scalar_1d)
    {
        check_convection_variable_velocity_scalar<1>();
    }

    TEST(fv_operators, convection_variable_velocity_scalar_2d)
    {
        check_convection_variable_velocity_scalar<2>();
    }

    TEST(fv_operators, convection_vector_exact_3d)
    {
        check_convection_exact_vector<3>();
    }

    // Identity returns the field unchanged, everywhere.
    template <std::size_t dim>
    void check_identity_exact()
    {
        const std::size_t level = 3;
        auto mesh               = uniform_mesh<dim>(level);
        auto u                  = make_scalar_field<double>("u", mesh);

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          double v = 1.;
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              v += static_cast<double>(d + 1) * cell.center(d) * cell.center(d);
                          }
                          u[cell] = v;
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        auto id     = make_identity<decltype(u)>();
        auto result = id(u);

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          EXPECT_NEAR(result[cell], u[cell], tol) << "dim=" << dim;
                      });
    }

    TEST(fv_operators, identity_exact_1d)
    {
        check_identity_exact<1>();
    }

    TEST(fv_operators, identity_exact_2d)
    {
        check_identity_exact<2>();
    }

    TEST(fv_operators, identity_exact_3d)
    {
        check_identity_exact<3>();
    }

    // The zero operator returns 0 everywhere, whatever the field.
    template <std::size_t dim>
    void check_zero_operator_exact()
    {
        const std::size_t level = 3;
        auto mesh               = uniform_mesh<dim>(level);
        auto u                  = make_scalar_field<double>("u", mesh);

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          double v = 1.;
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              v += static_cast<double>(d + 1) * cell.center(d);
                          }
                          u[cell] = v;
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        auto zero   = make_zero_operator<decltype(u)>();
        auto result = zero(u);

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          EXPECT_NEAR(result[cell], 0., tol) << "dim=" << dim;
                      });
    }

    TEST(fv_operators, zero_operator_exact_1d)
    {
        check_zero_operator_exact<1>();
    }

    TEST(fv_operators, zero_operator_exact_2d)
    {
        check_zero_operator_exact<2>();
    }

    TEST(fv_operators, zero_operator_exact_3d)
    {
        check_zero_operator_exact<3>();
    }

    // =====================================================================
    //  Convergence order on a smooth solution (uniform refinement)
    // =====================================================================

    namespace
    {
        // Observed order between two errors measured on meshes refined by 2.
        double observed_order(double err_coarse, double err_fine)
        {
            return std::log2(err_coarse / err_fine);
        }
    }

    // Diffusion is a 2nd-order scheme: the truncation error on a smooth field
    // decreases like h^2. We measure -Laplacian of a product of sines and check
    // the slope on interior cells (so the boundary condition plays no role).
    template <std::size_t dim>
    double diffusion_error(std::size_t level)
    {
        const std::size_t nx = 1u << level;
        auto mesh            = uniform_mesh<dim>(level);
        auto u               = make_scalar_field<double>("u", mesh);

        constexpr double k = 2 * M_PI;
        auto exact         = [&](const auto& x)
        {
            double v = 1.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                v *= std::sin(k * x(d));
            }
            return v;
        };
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = exact(cell.center());
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        DiffCoeff<dim> K;
        K.fill(1.);
        auto diff   = make_diffusion_order2<decltype(u)>(K);
        auto result = diff(u);

        // -Laplacian(prod sin) = dim * k^2 * prod sin
        double err = 0.;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 1))
                          {
                              double expected = dim * k * k * exact(cell.center());
                              err             = std::max(err, std::abs(result[cell] - expected));
                          }
                      });
        return err;
    }

    template <std::size_t dim>
    void check_diffusion_convergence()
    {
        double e4 = diffusion_error<dim>(4);
        double e5 = diffusion_error<dim>(5);
        double e6 = diffusion_error<dim>(6);

        double o1 = observed_order(e4, e5);
        double o2 = observed_order(e5, e6);

        EXPECT_NEAR(o1, 2.0, 0.2) << "dim=" << dim << " errors: " << e4 << " " << e5 << " " << e6;
        EXPECT_NEAR(o2, 2.0, 0.2) << "dim=" << dim << " errors: " << e4 << " " << e5 << " " << e6;
    }

    TEST(fv_operators, diffusion_convergence_1d)
    {
        check_diffusion_convergence<1>();
    }

    TEST(fv_operators, diffusion_convergence_2d)
    {
        check_diffusion_convergence<2>();
    }

    // WENO5 convection is 5th-order accurate on a smooth solution. We advect a
    // sine wave with a constant positive velocity and check the slope of the
    // error on interior cells (stencil half-width 3).
    double weno5_error_1d(std::size_t level)
    {
        const std::size_t nx = 1u << level;
        auto mesh            = uniform_mesh<1>(level, 6);
        auto u               = make_scalar_field<double>("u", mesh);

        constexpr double k = 2 * M_PI;
        auto exact         = [&](double x)
        {
            return std::sin(k * x);
        };
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = exact(cell.center(0));
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        VelocityVector<1> velocity;
        velocity(0) = 1.5;
        auto conv   = make_convection_weno5<decltype(u)>(velocity);
        auto result = conv(u);

        // v du/dx = v * k * cos(k x)
        double err = 0.;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (is_interior(cell, nx, 3))
                          {
                              double expected = velocity(0) * k * std::cos(k * cell.center(0));
                              err             = std::max(err, std::abs(result[cell] - expected));
                          }
                      });
        return err;
    }

    TEST(fv_operators, weno5_convergence_1d)
    {
        double e4 = weno5_error_1d(4);
        double e5 = weno5_error_1d(5);
        double e6 = weno5_error_1d(6);

        double o1 = observed_order(e4, e5);
        double o2 = observed_order(e5, e6);

        // WENO5 reaches 5th order on this smooth monotone-in-pieces solution; the
        // order can dip near the extrema, so we only require clearly above 4.
        EXPECT_GT(o1, 4.0) << "errors: " << e4 << " " << e5 << " " << e6;
        EXPECT_GT(o2, 4.0) << "errors: " << e4 << " " << e5 << " " << e6;
    }

    // =====================================================================
    //  Level-jump mesh (hand-built) : conservation and interface fluxes
    // =====================================================================
    //
    // This is the case that no demo checks in a verifiable way: a 2D mesh with a
    // coarse left half (level L) and a fine right half (level L+1), sharing a
    // vertical level-jump interface at x = 1/2. On such a mesh the schemes must:
    //   - return exactly 0 for every operator on a constant field
    //     (conservation: the fluxes cancel, including across the jump);
    //   - stay exact for a linear field where the operator is exact on a uniform
    //     mesh (upwind convection, gradient, divergence, and the null diffusion).

    namespace
    {
        constexpr std::size_t coarse_level = 3;
        constexpr std::size_t fine_level   = 4;

        // Coarse left half (x in [0,1/2]) at coarse_level, fine right half
        // (x in [1/2,1]) at fine_level, full extent in y. Unit box [0,1]^2.
        auto two_level_mesh_2d()
        {
            constexpr std::size_t dim = 2;
            auto cfg                  = mesh_config<dim>().min_level(coarse_level).max_level(fine_level);
            using Mesh                = decltype(mra::make_empty_mesh(cfg));
            using cl_t                = typename Mesh::cl_type;

            const int nc = 1 << coarse_level; // 8
            const int nf = 1 << fine_level;   // 16

            cl_t cl;
            for (int y = 0; y < nc; ++y)
            {
                cl[coarse_level][{y}].add_interval({0, nc / 2}); // x-index 0..3
            }
            for (int y = 0; y < nf; ++y)
            {
                cl[fine_level][{y}].add_interval({nf / 2, nf}); // x-index 8..15
            }
            return mra::make_mesh(cl, cfg);
        }

        // A cell touches the outer domain boundary if, at its own level, one of its
        // indices is the first or last cell of the unit box. Those cells depend on
        // the boundary condition and are excluded from the linear-field checks.
        template <class Cell>
        bool touches_domain_boundary(const Cell& cell)
        {
            const auto n = static_cast<long long>(1u << cell.level);
            for (std::size_t d = 0; d < Cell::dim; ++d)
            {
                const auto i = static_cast<long long>(cell.indices[d]);
                if (i == 0 || i == n - 1)
                {
                    return true;
                }
            }
            return false;
        }
    }

    // Conservation: every operator vanishes on a constant field, everywhere
    // (including the level-jump interface and the boundary, since the Dirichlet
    // value matches the constant).
    TEST(fv_operators, level_jump_constant_conservation)
    {
        constexpr double c = 2.5;

        auto mesh = two_level_mesh_2d();

        auto scalar = make_scalar_field<double>("s", mesh);
        scalar.fill(c);
        make_bc<Dirichlet<1>>(scalar, c);

        auto vec = make_vector_field<double, 2>("v", mesh);
        vec.fill(c);
        make_bc<Dirichlet<1>>(vec, c, c);

        DiffCoeff<2> K;
        K.fill(1.);
        VelocityVector<2> velocity = {1., 2.};

        auto diff = make_diffusion_order2<decltype(scalar)>(K);
        auto conv = make_convection_upwind<decltype(scalar)>(velocity);
        auto grad = make_gradient_order2<decltype(scalar)>();
        auto div  = make_divergence_order2<decltype(vec)>();

        auto rd = diff(scalar);
        auto rc = conv(scalar);
        auto rg = grad(scalar);
        auto rv = div(vec);

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          EXPECT_NEAR(rd[cell], 0., tol) << "diffusion, level " << cell.level;
                          EXPECT_NEAR(rc[cell], 0., tol) << "convection, level " << cell.level;
                          EXPECT_NEAR(rg[cell](0), 0., tol) << "gradient x";
                          EXPECT_NEAR(rg[cell](1), 0., tol) << "gradient y";
                          EXPECT_NEAR(rv[cell], 0., tol) << "divergence, level " << cell.level;
                      });
    }

    // Interface fluxes: on a linear field the *central-flux* operators stay exact
    // across the level jump (the ghost cells filled by projection/prediction
    // reproduce a linear field exactly, and the averaged face flux is exact for a
    // linear field). This covers gradient, divergence, and the null diffusion.
    //
    // Upwind convection is deliberately NOT checked here: at a level jump the
    // finer-side flux is evaluated at the fine face position (x_face - h_fine/2),
    // shifted from the coarse cell center, so the one-sided upwind difference is
    // no longer exact for a linear field (upwind is only 1st order there). Its
    // level-jump behaviour is still covered by the constant-field conservation
    // test above.
    //
    // Boundary-touching cells are excluded (BC-dependent).
    TEST(fv_operators, level_jump_linear_exactness)
    {
        // u(x,y) = a x + b y + c   (scalar) ; components share the same shape.
        constexpr double a = 3., b = -2., c = 1.;

        auto mesh = two_level_mesh_2d();

        auto u = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = a * cell.center(0) + b * cell.center(1) + c;
                      });
        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto& coord)
                              {
                                  return a * coord[0] + b * coord[1] + c;
                              });

        auto vecf = make_vector_field<double, 2>("uv", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          vecf[cell](0) = a * cell.center(0) + b * cell.center(1) + c;
                          vecf[cell](1) = b * cell.center(0) + a * cell.center(1) - c;
                      });
        make_bc<Dirichlet<1>>(vecf,
                              [](const auto&, const auto&, const auto& coord)
                              {
                                  Array<double, 2> v;
                                  v(0) = a * coord[0] + b * coord[1] + c;
                                  v(1) = b * coord[0] + a * coord[1] - c;
                                  return v;
                              });

        auto diff = make_diffusion_order2<decltype(u)>(DiffCoeff<2>{1., 1.});
        auto grad = make_gradient_order2<decltype(u)>();
        auto div  = make_divergence_order2<decltype(vecf)>();

        auto rd = diff(u);
        auto rg = grad(u);
        auto rv = div(vecf);

        const double div_expected = a + a; // du0/dx + du1/dy = a + a

        std::size_t nb_checked = 0;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          if (touches_domain_boundary(cell))
                          {
                              return;
                          }
                          EXPECT_NEAR(rd[cell], 0., 1e-9) << "diffusion(linear), level " << cell.level;
                          EXPECT_NEAR(rg[cell](0), a, 1e-9) << "gradient x, level " << cell.level;
                          EXPECT_NEAR(rg[cell](1), b, 1e-9) << "gradient y, level " << cell.level;
                          EXPECT_NEAR(rv[cell], div_expected, 1e-9) << "divergence, level " << cell.level;
                          ++nb_checked;
                      });
        EXPECT_GT(nb_checked, 0u);
    }

#ifdef SAMURAI_WITH_PETSC
    // =====================================================================
    //  Explicit / implicit consistency (PETSc)
    // =====================================================================
    //
    // Every operator has two consumers: the explicit application op(u) and the
    // matrix assembled by petsc/. On interior cells (whose stencil never reaches a
    // ghost, so no boundary-condition folding takes place) the matrix-vector
    // product A*u must reproduce op(u) to round-off. This covers scalar and vector
    // operators (rectangular ones too, e.g. divergence), validating the assembly's
    // scalar/matrix coefficient handling on the same footing as the explicit path.

    namespace
    {
        // op(u) (explicit) vs A*u (assembled), compared on interior cells for every
        // output component. `u` must already carry a boundary condition.
        //
        // The explicit result is snapshotted into a plain std::vector before the
        // matrix is applied, so the comparison holds no matter how fields copy.
        template <class Scheme, class InputField>
        void expect_explicit_matches_implicit(Scheme& scheme, InputField& u, std::size_t nx, std::size_t half_width)
        {
            auto expl         = scheme(u); // explicit; also fills u's ghosts through the BC
            using out_field_t = decltype(expl);

            std::vector<double> explicit_values;
            for_each_cell(u.mesh(),
                          [&](const auto& cell)
                          {
                              if (is_interior(cell, nx, half_width))
                              {
                                  if constexpr (out_field_t::is_scalar)
                                  {
                                      explicit_values.push_back(expl[cell]);
                                  }
                                  else
                                  {
                                      for (std::size_t c = 0; c < out_field_t::n_comp; ++c)
                                      {
                                          explicit_values.push_back(expl[cell](c));
                                      }
                                  }
                              }
                          });

            auto impl = expl; // same (output) field type; receives A*u
            impl.fill(0.);

            auto assembly = petsc::make_assembly(scheme);
            assembly.set_unknown(u);
            Mat A;
            assembly.create_matrix(A);
            assembly.assemble_matrix(A);

            Vec x = petsc::create_petsc_vector_from(u);    // input unknowns (columns)
            Vec y = petsc::create_petsc_vector_from(impl); // output values (rows), aliases impl
            MatMult(A, x, y);

            std::size_t k          = 0;
            std::size_t nb_checked = 0;
            for_each_cell(u.mesh(),
                          [&](const auto& cell)
                          {
                              if (is_interior(cell, nx, half_width))
                              {
                                  if constexpr (out_field_t::is_scalar)
                                  {
                                      EXPECT_NEAR(impl[cell], explicit_values[k++], 1e-10);
                                  }
                                  else
                                  {
                                      for (std::size_t c = 0; c < out_field_t::n_comp; ++c)
                                      {
                                          EXPECT_NEAR(impl[cell](c), explicit_values[k++], 1e-10) << "c=" << c;
                                      }
                                  }
                                  ++nb_checked;
                              }
                          });
            EXPECT_GT(nb_checked, 0u);

            VecDestroy(&x);
            VecDestroy(&y);
            MatDestroy(&A);
        }
    }

    TEST(fv_operators, explicit_vs_implicit_diffusion)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;
        const std::size_t nx      = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          auto x  = cell.center();
                          u[cell] = x(0) * x(0) + 2 * x(1) * x(1) + x(0) * x(1) + 0.5;
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        DiffCoeff<dim> K;
        K.fill(1.);
        auto diff = make_diffusion_order2<decltype(u)>(K);
        expect_explicit_matches_implicit(diff, u, nx, 1);
    }

    TEST(fv_operators, explicit_vs_implicit_diffusion_vector)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;
        const std::size_t nx      = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_vector_field<double, dim>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          auto x = cell.center();
                          for (std::size_t c = 0; c < dim; ++c)
                          {
                              u[cell](c) = static_cast<double>(c + 1) * x(0) * x(0) + x(1) * x(1) + static_cast<double>(c) * x(0) * x(1);
                          }
                      });
        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> zero;
                                  zero.fill(0.);
                                  return zero;
                              });

        DiffCoeff<dim> K;
        K.fill(1.);
        auto diff = make_diffusion_order2<decltype(u)>(K);
        expect_explicit_matches_implicit(diff, u, nx, 1);
    }

    TEST(fv_operators, explicit_vs_implicit_divergence)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;
        const std::size_t nx      = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_vector_field<double, dim>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          auto x = cell.center();
                          for (std::size_t c = 0; c < dim; ++c)
                          {
                              u[cell](c) = static_cast<double>(c + 1) * x(0) + static_cast<double>(c + 2) * x(1);
                          }
                      });
        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> zero;
                                  zero.fill(0.);
                                  return zero;
                              });

        auto div = make_divergence_order2<decltype(u)>();
        expect_explicit_matches_implicit(div, u, nx, 1);
    }

    TEST(fv_operators, explicit_vs_implicit_convection)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;
        const std::size_t nx      = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          auto x  = cell.center();
                          u[cell] = 2 * x(0) + 3 * x(1) + 0.5;
                      });
        make_bc<Dirichlet<1>>(u, 0.);

        VelocityVector<dim> velocity = {1., 2.};
        auto conv                    = make_convection_upwind<decltype(u)>(velocity);
        expect_explicit_matches_implicit(conv, u, nx, 1);
    }

    TEST(fv_operators, explicit_vs_implicit_convection_vector)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;
        const std::size_t nx      = 1u << level;

        auto mesh = uniform_mesh<dim>(level);
        auto u    = make_vector_field<double, dim>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          auto x = cell.center();
                          for (std::size_t c = 0; c < dim; ++c)
                          {
                              u[cell](c) = static_cast<double>(c + 1) * x(0) + static_cast<double>(c + 2) * x(1);
                          }
                      });
        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  Array<double, dim> zero;
                                  zero.fill(0.);
                                  return zero;
                              });

        VelocityVector<dim> velocity = {1., 2.};
        auto conv                    = make_convection_upwind<decltype(u)>(velocity);
        expect_explicit_matches_implicit(conv, u, nx, 1);
    }
#endif
}
