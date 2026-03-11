// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/numeric/prediction.hpp>
#include <samurai/numeric/projection.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    namespace
    {
        //=========================================================================
        // Helper: mesh creation
        //
        // Creates a uniform MR mesh with min_level = L-1, max_level = L,
        // so that all cells begin at the fine level L.
        // Generic for any dim >= 1: builds the unit hypercube [0,1]^dim.
        //=========================================================================
        template <std::size_t dim, int s>
        auto make_two_level_mesh(std::size_t L)
        {
            auto mesh_cfg = mesh_config<dim, s>().min_level(L - 1).max_level(L).max_stencil_radius(std::max(2, s)).disable_args_parse();
            // Build unit-hypercube box generically: [0,1]^dim
            using point_t = xt::xtensor_fixed<double, xt::xshape<dim>>;
            point_t lo, hi;
            lo.fill(0.0);
            hi.fill(1.0);
            auto box = Box<double, dim>(lo, hi);
            return mra::make_mesh(box, mesh_cfg);
        }

        //=========================================================================
        // Helper: 1D cell average
        //
        // Computes (1/h) * integral_{xc-h/2}^{xc+h/2} p(x) dx using the
        // antiderivative of p.  Exact in floating-point for polynomial antiderivs.
        //=========================================================================
        template <class Cell, class AntiderivFn>
        double cell_avg_1d(const Cell& cell, const AntiderivFn& antideriv)
        {
            const double xc   = cell.center()[0];
            const double half = cell.length / 2.0;
            return (antideriv(xc + half) - antideriv(xc - half)) / cell.length;
        }

        //=========================================================================
        // Helper: 2D cell average for separable polynomial f(x)*g(y)
        //
        // avg = avg_x(f) * avg_y(g), computed via antiderivatives.
        //=========================================================================
        template <class Cell, class AntiderivX, class AntiderivY>
        double cell_avg_2d(const Cell& cell, const AntiderivX& antideriv_x, const AntiderivY& antideriv_y)
        {
            const double xc   = cell.center()[0];
            const double yc   = cell.center()[1];
            const double half = cell.length / 2.0;
            const double h    = cell.length;
            const double ax   = (antideriv_x(xc + half) - antideriv_x(xc - half)) / h;
            const double ay   = (antideriv_y(yc + half) - antideriv_y(yc - half)) / h;
            return ax * ay;
        }

        //=========================================================================
        // Helper: N-dimensional cell average for separable polynomial
        //
        // Given a cell and a callable antiderivs(i) -> antideriv function for
        // axis i, computes product_{i=0}^{dim-1} avg_i(antiderivs(i)).
        // Separability means cell_avg(f) = prod_i cell_avg_1d(f_i).
        //=========================================================================
        template <std::size_t dim, class Cell, class AntiderivArray>
        double cell_avg_nd(const Cell& cell, const AntiderivArray& antiderivs)
        {
            const double half = cell.length / 2.0;
            const double h    = cell.length;
            double result     = 1.0;
            for (std::size_t i = 0; i < dim; ++i)
            {
                const double ci = cell.center()[i];
                result *= (antiderivs(i)(ci + half) - antiderivs(i)(ci - half)) / h;
            }
            return result;
        }

        //=========================================================================
        // Helper: error calculation — scalar field
        //
        // Returns the maximum absolute difference between field values at level L
        // and the reference cell-average function applied to each cell.
        //=========================================================================
        template <class Field, class CellAvgFn>
        double max_abs_error(const Field& u, std::size_t L, const CellAvgFn& cell_avg_fn)
        {
            using mesh_id_t = typename Field::mesh_t::mesh_id_t;
            double max_err  = 0.0;
            for_each_cell(u.mesh()[mesh_id_t::cells][L],
                          [&](const auto& cell)
                          {
                              const double expected = cell_avg_fn(cell);
                              const double actual   = u[cell];
                              const double err      = std::abs(actual - expected);
                              max_err               = std::max(max_err, err);
                          });
            return max_err;
        }

        //=========================================================================
        // Helper: error calculation — vector field
        //
        // Checks every component c: expected value is (c+1) * cell_avg_fn(cell).
        // Returns the maximum absolute difference over all cells and components.
        //=========================================================================
        template <std::size_t n_comp, class Field, class CellAvgFn>
        double max_abs_error_vector(const Field& u, std::size_t L, const CellAvgFn& cell_avg_fn)
        {
            using mesh_id_t = typename Field::mesh_t::mesh_id_t;
            double max_err  = 0.0;
            for_each_cell(u.mesh()[mesh_id_t::cells][L],
                          [&](const auto& cell)
                          {
                              const double base = cell_avg_fn(cell);
                              for (std::size_t c = 0; c < n_comp; ++c)
                              {
                                  const double expected = static_cast<double>(c + 1) * base;
                                  const double actual   = u[cell][static_cast<int>(c)];
                                  const double err      = std::abs(actual - expected);
                                  max_err               = std::max(max_err, err);
                              }
                          });
            return max_err;
        }

        //=========================================================================
        // Helper: roundtrip test body
        //
        // 1. Build uniform mesh (levels L-1 and L).
        // 2. Fill ALL cells and ghosts at every level with exact cell averages.
        //    Ghost cells at L-1 need exact values so that the prediction stencil
        //    (which reads s coarse neighbors on each side) operates correctly.
        // 3. Project level L -> L-1 (overwrites interior L-1 cells with cell averages
        //    computed from fine children — exact for any polynomial).
        // 4. Predict level L-1 -> L  (exact for polynomials of degree < 2s+1).
        // 5. Return max absolute error vs. exact cell averages at level L.
        //
        // Subset algebra pattern (Design Decision 4):
        //   projection:  intersection(mesh[cells][L], mesh[reference][L-1]).on(L-1)
        //   prediction:  intersection(mesh[cells][L], mesh[reference][L-1]).on(L)
        //=========================================================================
        template <std::size_t s, std::size_t dim, class CellAvgFn>
        double roundtrip_error(std::size_t L, const CellAvgFn& cell_avg_fn)
        {
            using mesh_id_t = typename decltype(make_two_level_mesh<dim, static_cast<int>(s)>(L))::mesh_id_t;

            auto mesh = make_two_level_mesh<dim, static_cast<int>(s)>(L);

            auto u = make_scalar_field<double>("u", mesh);
            u.resize();

            // Initialize ALL cells and ghosts at every level with exact cell averages.
            // Ghost cells at L-1 must hold correct values so that the prediction stencil
            // has valid coarse-level neighbors at the domain boundary.
            for (std::size_t lvl = mesh.min_level(); lvl <= mesh.max_level(); ++lvl)
            {
                for_each_cell(mesh[mesh_id_t::reference][lvl],
                              [&](const auto& cell)
                              {
                                  u[cell] = cell_avg_fn(cell);
                              });
            }

            // Project: level L -> L-1 (average fine children to get coarse cell average)
            auto proj_set = intersection(mesh[mesh_id_t::cells][L], mesh[mesh_id_t::reference][L - 1]).on(L - 1);
            proj_set.apply_op(projection(u));

            // Predict: level L-1 -> L  (dest_on_level=false: reads from level-1, writes to level)
            auto pred_set = intersection(mesh[mesh_id_t::cells][L], mesh[mesh_id_t::reference][L - 1]).on(L);
            pred_set.apply_op(prediction<s, false>(u));

            return max_abs_error(u, L, cell_avg_fn);
        }

        //=========================================================================
        // Helper: VectorField roundtrip
        //
        // Same logic as roundtrip_error, but uses a VectorField with n_comp
        // components. Component c is initialised with (c+1) * cell_avg_fn(cell),
        // so each component carries a distinct non-trivial polynomial and we verify
        // all components are recovered after projection + prediction.
        //=========================================================================
        template <std::size_t n_comp, std::size_t s, std::size_t dim, class CellAvgFn>
        double roundtrip_error_vector(std::size_t L, const CellAvgFn& cell_avg_fn)
        {
            using mesh_id_t = typename decltype(make_two_level_mesh<dim, static_cast<int>(s)>(L))::mesh_id_t;

            auto mesh = make_two_level_mesh<dim, static_cast<int>(s)>(L);

            auto u = make_vector_field<double, n_comp>("u", mesh);
            u.resize();

            // Initialize ALL cells and ghosts: component c gets (c+1) * cell_avg_fn(cell)
            for (std::size_t lvl = mesh.min_level(); lvl <= mesh.max_level(); ++lvl)
            {
                for_each_cell(mesh[mesh_id_t::reference][lvl],
                              [&](const auto& cell)
                              {
                                  const double base = cell_avg_fn(cell);
                                  for (std::size_t c = 0; c < n_comp; ++c)
                                  {
                                      u[cell][static_cast<int>(c)] = static_cast<double>(c + 1) * base;
                                  }
                              });
            }

            // Project: level L -> L-1
            auto proj_set = intersection(mesh[mesh_id_t::cells][L], mesh[mesh_id_t::reference][L - 1]).on(L - 1);
            proj_set.apply_op(projection(u));

            // Predict: level L-1 -> L
            auto pred_set = intersection(mesh[mesh_id_t::cells][L], mesh[mesh_id_t::reference][L - 1]).on(L);
            pred_set.apply_op(prediction<s, false>(u));

            return max_abs_error_vector<n_comp>(u, L, cell_avg_fn);
        }

    } // anonymous namespace

    //=========================================================================
    // Stencil radius 0 — constant polynomials
    // Prediction order 1: roundtrip is exact for degree 0 (constants).
    // Cell average of a constant equals the constant itself.
    //=========================================================================

    TEST(ProjectionPredictionRoundtrip, s0_1D_constant)
    {
        // u(x) = 3.14  — cell average equals 3.14 at all cells
        const double err = roundtrip_error<0, 1>(4,
                                                 [](const auto& /*cell*/)
                                                 {
                                                     return 3.14;
                                                 });
        EXPECT_LE(err, 1e-14) << "1D constant polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s0_2D_constant)
    {
        const double err = roundtrip_error<0, 2>(4,
                                                 [](const auto& /*cell*/)
                                                 {
                                                     return 3.14;
                                                 });
        EXPECT_LE(err, 1e-14) << "2D constant polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s0_3D_constant)
    {
        const double err = roundtrip_error<0, 3>(4,
                                                 [](const auto& /*cell*/)
                                                 {
                                                     return 3.14;
                                                 });
        EXPECT_LE(err, 1e-14) << "3D constant polynomial roundtrip error too large: " << err;
    }

    //=========================================================================
    // Stencil radius 1 — polynomials up to degree 2
    // Prediction order 3: roundtrip is exact for cell averages of degree < 3.
    //
    // Fields store cell averages; we initialize using exact antiderivative
    // integration so that the data is consistent with the FV interpretation.
    //=========================================================================

    TEST(ProjectionPredictionRoundtrip, s1_1D_linear)
    {
        // u(x) = 1 + 2x,  antideriv P(x) = x + x^2
        // Cell average over [xc-h/2, xc+h/2] = 1 + 2*xc  (no O(h^2) correction for linears)
        const double err = roundtrip_error<1, 1>(4,
                                                 [](const auto& cell)
                                                 {
                                                     return cell_avg_1d(cell,
                                                                        [](double x)
                                                                        {
                                                                            const double x2 = x * x;
                                                                            return x + x2; // P(x) = x + x^2  (antideriv of 1+2x)
                                                                        });
                                                 });
        EXPECT_LE(err, 1e-13) << "1D linear polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s1_1D_quadratic)
    {
        // u(x) = 1 + 2x + 3x^2,  antideriv P(x) = x + x^2 + x^3
        // Cell avg = 1 + 2*xc + 3*(xc^2 + h^2/12)  — differs from point value at center
        const double err = roundtrip_error<1, 1>(4,
                                                 [](const auto& cell)
                                                 {
                                                     return cell_avg_1d(cell,
                                                                        [](double x)
                                                                        {
                                                                            const double x2 = x * x;
                                                                            const double x3 = x2 * x;
                                                                            return x + x2 + x3; // antideriv of (1 + 2x + 3x^2)
                                                                        });
                                                 });
        EXPECT_LE(err, 1e-13) << "1D quadratic polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s1_2D_degree2)
    {
        // u(x,y) = (1+x)*(1+y)  — separable, degree 2
        // Antiderivs: Ax(x) = x + x^2/2,  Ay(y) = y + y^2/2
        const double err = roundtrip_error<1, 2>(4,
                                                 [](const auto& cell)
                                                 {
                                                     return cell_avg_2d(
                                                         cell,
                                                         [](double x)
                                                         {
                                                             const double x2 = x * x;
                                                             return x + (x2 / 2.0); // antideriv of (1+x)
                                                         },
                                                         [](double y)
                                                         {
                                                             const double y2 = y * y;
                                                             return y + (y2 / 2.0); // antideriv of (1+y)
                                                         });
                                                 });
        EXPECT_LE(err, 1e-13) << "2D degree-2 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s1_3D_degree2)
    {
        // u(x,y,z) = (1+x)*(1+y)*(1+z)  — fully separable, degree 3 (each factor linear)
        // Antideriv for each: A(t) = t + t^2/2
        const double err = roundtrip_error<1, 3>(4,
                                                 [](const auto& cell)
                                                 {
                                                     const double xc   = cell.center()[0];
                                                     const double yc   = cell.center()[1];
                                                     const double zc   = cell.center()[2];
                                                     const double half = cell.length / 2.0;
                                                     const double h    = cell.length;
                                                     auto antideriv    = [](double t)
                                                     {
                                                         const double t2 = t * t;
                                                         return t + (t2 / 2.0);
                                                     };
                                                     const double ax = (antideriv(xc + half) - antideriv(xc - half)) / h;
                                                     const double ay = (antideriv(yc + half) - antideriv(yc - half)) / h;
                                                     const double az = (antideriv(zc + half) - antideriv(zc - half)) / h;
                                                     return ax * ay * az;
                                                 });
        EXPECT_LE(err, 1e-13) << "3D degree-2 polynomial roundtrip error too large: " << err;
    }

    //=========================================================================
    // Stencil radius 2 — polynomials up to degree 4
    // Prediction order 5: roundtrip is exact for cell averages of degree < 5.
    //
    // Separable polynomials: cell avg factors as avg_x(f) * avg_y(g) * ...
    //=========================================================================

    TEST(ProjectionPredictionRoundtrip, s2_1D_degree4)
    {
        // u(x) = 1 + x + x^2 + x^3 + x^4
        // Antideriv P(x) = x + x^2/2 + x^3/3 + x^4/4 + x^5/5
        const double err = roundtrip_error<2, 1>(4,
                                                 [](const auto& cell)
                                                 {
                                                     return cell_avg_1d(cell,
                                                                        [](double x)
                                                                        {
                                                                            const double x2 = x * x;
                                                                            const double x3 = x2 * x;
                                                                            const double x4 = x3 * x;
                                                                            const double x5 = x4 * x;
                                                                            return x + (x2 / 2.0) + (x3 / 3.0) + (x4 / 4.0) + (x5 / 5.0);
                                                                        });
                                                 });
        EXPECT_LE(err, 1e-13) << "1D degree-4 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s2_2D_degree4)
    {
        // u(x,y) = (1 + x + x^2) * (1 + y + y^2) — separable, max degree per variable = 2, total = 4
        // Antideriv_x: Ax(x) = x + x^2/2 + x^3/3;  Antideriv_y: Ay(y) = y + y^2/2 + y^3/3
        const double err = roundtrip_error<2, 2>(4,
                                                 [](const auto& cell)
                                                 {
                                                     return cell_avg_2d(
                                                         cell,
                                                         [](double x)
                                                         {
                                                             const double x2 = x * x;
                                                             const double x3 = x2 * x;
                                                             return x + (x2 / 2.0) + (x3 / 3.0); // antideriv of (1+x+x^2)
                                                         },
                                                         [](double y)
                                                         {
                                                             const double y2 = y * y;
                                                             const double y3 = y2 * y;
                                                             return y + (y2 / 2.0) + (y3 / 3.0); // antideriv of (1+y+y^2)
                                                         });
                                                 });
        EXPECT_LE(err, 1e-13) << "2D degree-4 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s2_3D_degree4)
    {
        // u(x,y,z) = (1+x+x^2)*(1+y)*(1+z) — separable, degree 4
        const double err = roundtrip_error<2, 3>(4,
                                                 [](const auto& cell)
                                                 {
                                                     const double xc   = cell.center()[0];
                                                     const double yc   = cell.center()[1];
                                                     const double zc   = cell.center()[2];
                                                     const double half = cell.length / 2.0;
                                                     const double h    = cell.length;
                                                     auto antideriv_x  = [](double x)
                                                     {
                                                         const double x2 = x * x;
                                                         const double x3 = x2 * x;
                                                         return x + (x2 / 2.0) + (x3 / 3.0);
                                                     };
                                                     auto antideriv_yz = [](double t)
                                                     {
                                                         const double t2 = t * t;
                                                         return t + (t2 / 2.0);
                                                     };
                                                     const double ax = (antideriv_x(xc + half) - antideriv_x(xc - half)) / h;
                                                     const double ay = (antideriv_yz(yc + half) - antideriv_yz(yc - half)) / h;
                                                     const double az = (antideriv_yz(zc + half) - antideriv_yz(zc - half)) / h;
                                                     return ax * ay * az;
                                                 });
        EXPECT_LE(err, 1e-13) << "3D degree-4 polynomial roundtrip error too large: " << err;
    }

    //=========================================================================
    // Stencil radius 3 — polynomials up to degree 6
    // Prediction order 7: roundtrip is exact for cell averages of degree < 7.
    //=========================================================================

    TEST(ProjectionPredictionRoundtrip, s3_1D_degree6)
    {
        // u(x) = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6
        // Antideriv P(x) = x + x^2/2 + x^3/3 + x^4/4 + x^5/5 + x^6/6 + x^7/7
        const double err = roundtrip_error<3, 1>(5,
                                                 [](const auto& cell)
                                                 {
                                                     return cell_avg_1d(cell,
                                                                        [](double x)
                                                                        {
                                                                            const double x2 = x * x;
                                                                            const double x3 = x2 * x;
                                                                            const double x4 = x3 * x;
                                                                            const double x5 = x4 * x;
                                                                            const double x6 = x5 * x;
                                                                            const double x7 = x6 * x;
                                                                            return x + (x2 / 2.0) + (x3 / 3.0) + (x4 / 4.0) + (x5 / 5.0)
                                                                                 + (x6 / 6.0) + (x7 / 7.0);
                                                                        });
                                                 });
        EXPECT_LE(err, 1e-13) << "1D degree-6 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s3_2D_degree6)
    {
        // u(x,y) = (1 + x + x^2 + x^3) * (1 + y + y^2 + y^3) — separable, degree 6
        // Antiderivs: A(t) = t + t^2/2 + t^3/3 + t^4/4
        const double err = roundtrip_error<3, 2>(5,
                                                 [](const auto& cell)
                                                 {
                                                     return cell_avg_2d(
                                                         cell,
                                                         [](double x)
                                                         {
                                                             const double x2 = x * x;
                                                             const double x3 = x2 * x;
                                                             const double x4 = x3 * x;
                                                             return x + (x2 / 2.0) + (x3 / 3.0) + (x4 / 4.0);
                                                         },
                                                         [](double y)
                                                         {
                                                             const double y2 = y * y;
                                                             const double y3 = y2 * y;
                                                             const double y4 = y3 * y;
                                                             return y + (y2 / 2.0) + (y3 / 3.0) + (y4 / 4.0);
                                                         });
                                                 });
        EXPECT_LE(err, 1e-13) << "2D degree-6 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s3_3D_degree6)
    {
        // u(x,y,z) = (1+x+x^2+x^3)*(1+y+y^2)*(1+z+z^2) — separable, degree 7 per-variable max 3,2,2
        // Each variable's max degree is within prediction exactness range for s=3.
        const double err = roundtrip_error<3, 3>(5,
                                                 [](const auto& cell)
                                                 {
                                                     const double xc   = cell.center()[0];
                                                     const double yc   = cell.center()[1];
                                                     const double zc   = cell.center()[2];
                                                     const double half = cell.length / 2.0;
                                                     const double h    = cell.length;
                                                     auto antideriv_x  = [](double x)
                                                     {
                                                         const double x2 = x * x;
                                                         const double x3 = x2 * x;
                                                         const double x4 = x3 * x;
                                                         return x + (x2 / 2.0) + (x3 / 3.0) + (x4 / 4.0);
                                                     };
                                                     auto antideriv_yz = [](double t)
                                                     {
                                                         const double t2 = t * t;
                                                         const double t3 = t2 * t;
                                                         return t + (t2 / 2.0) + (t3 / 3.0);
                                                     };
                                                     const double ax = (antideriv_x(xc + half) - antideriv_x(xc - half)) / h;
                                                     const double ay = (antideriv_yz(yc + half) - antideriv_yz(yc - half)) / h;
                                                     const double az = (antideriv_yz(zc + half) - antideriv_yz(zc - half)) / h;
                                                     return ax * ay * az;
                                                 });
        // 3D high-order case: allow slightly larger tolerance due to floating-point accumulation
        EXPECT_LE(err, 5e-13) << "3D degree-6 polynomial roundtrip error too large: " << err;
    }

    //=========================================================================
    // 4D, 5D, 6D tests
    //
    // For high-dimensional cases we use L=3 (8 fine cells per axis) to keep
    // compilation and runtime manageable.  All polynomials are separable so
    // that the exact cell-average factors per axis.
    //=========================================================================

    // --- s=0: constants ---

    TEST(ProjectionPredictionRoundtrip, s0_4D_constant)
    {
        const double err = roundtrip_error<0, 4>(3,
                                                 [](const auto& /*cell*/)
                                                 {
                                                     return 3.14;
                                                 });
        EXPECT_LE(err, 1e-14) << "4D constant polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s0_5D_constant)
    {
        const double err = roundtrip_error<0, 5>(3,
                                                 [](const auto& /*cell*/)
                                                 {
                                                     return 3.14;
                                                 });
        EXPECT_LE(err, 1e-14) << "5D constant polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s0_6D_constant)
    {
        const double err = roundtrip_error<0, 6>(3,
                                                 [](const auto& /*cell*/)
                                                 {
                                                     return 3.14;
                                                 });
        EXPECT_LE(err, 1e-14) << "6D constant polynomial roundtrip error too large: " << err;
    }

    // --- s=1: degree-2 separable polynomial (each axis factor: 1+t, antideriv: t+t^2/2) ---

    TEST(ProjectionPredictionRoundtrip, s1_4D_degree2)
    {
        // u = prod_{i=0}^{3} (1 + x_i),  antideriv per axis: A(t) = t + t^2/2
        auto antideriv = [](double t)
        {
            return t + (t * t / 2.0);
        };
        const double err = roundtrip_error<1, 4>(3,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<4>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 1e-13) << "4D degree-2 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s1_5D_degree2)
    {
        auto antideriv = [](double t)
        {
            return t + (t * t / 2.0);
        };
        const double err = roundtrip_error<1, 5>(3,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<5>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 1e-13) << "5D degree-2 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s1_6D_degree2)
    {
        auto antideriv = [](double t)
        {
            return t + (t * t / 2.0);
        };
        const double err = roundtrip_error<1, 6>(3,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<6>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 1e-13) << "6D degree-2 polynomial roundtrip error too large: " << err;
    }

    // --- s=2: degree-4 separable polynomial (each axis: 1+t+t^2, antideriv: t+t^2/2+t^3/3) ---

    TEST(ProjectionPredictionRoundtrip, s2_4D_degree4)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            return t + (t2 / 2.0) + (t3 / 3.0);
        };
        const double err = roundtrip_error<2, 4>(3,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<4>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 5e-13) << "4D degree-4 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s2_5D_degree4)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            return t + (t2 / 2.0) + (t3 / 3.0);
        };
        const double err = roundtrip_error<2, 5>(2,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<5>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 5e-12) << "5D degree-4 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s2_6D_degree4)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            return t + (t2 / 2.0) + (t3 / 3.0);
        };
        const double err = roundtrip_error<2, 6>(2,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<6>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 5e-11) << "6D degree-4 polynomial roundtrip error too large: " << err;
    }

    // --- s=3: degree-6 separable polynomial (each axis: 1+t+t^2+t^3, antideriv: t+t^2/2+t^3/3+t^4/4) ---

    TEST(ProjectionPredictionRoundtrip, s3_4D_degree6)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            return t + (t2 / 2.0) + (t3 / 3.0) + (t4 / 4.0);
        };
        const double err = roundtrip_error<3, 4>(3,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<4>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 2e-12) << "4D degree-6 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s3_5D_degree6)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            return t + (t2 / 2.0) + (t3 / 3.0) + (t4 / 4.0);
        };
        const double err = roundtrip_error<3, 5>(2,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<5>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 1e-11) << "5D degree-6 polynomial roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, s3_6D_degree6)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            return t + (t2 / 2.0) + (t3 / 3.0) + (t4 / 4.0);
        };
        const double err = roundtrip_error<3, 6>(2,
                                                 [&antideriv](const auto& cell)
                                                 {
                                                     return cell_avg_nd<6>(cell,
                                                                           [&antideriv](std::size_t /*i*/)
                                                                           {
                                                                               return antideriv;
                                                                           });
                                                 });
        EXPECT_LE(err, 5e-10) << "6D degree-6 polynomial roundtrip error too large: " << err;
    }

    //=========================================================================
    // VectorField roundtrip tests
    //
    // Mirror of the scalar tests above using a 2-component VectorField.
    // Component 0 carries 1*cell_avg, component 1 carries 2*cell_avg,
    // so both components carry distinct non-trivial data and we verify
    // all are recovered after projection + prediction.
    //
    // Tolerances and levels are identical to the corresponding scalar tests.
    //=========================================================================

    // --- s=0: constants ---

    TEST(ProjectionPredictionRoundtrip, vec2_s0_1D_constant)
    {
        const double err = roundtrip_error_vector<2, 0, 1>(4,
                                                           [](const auto& /*cell*/)
                                                           {
                                                               return 3.14;
                                                           });
        EXPECT_LE(err, 1e-14) << "vec2 1D constant roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s0_2D_constant)
    {
        const double err = roundtrip_error_vector<2, 0, 2>(4,
                                                           [](const auto& /*cell*/)
                                                           {
                                                               return 3.14;
                                                           });
        EXPECT_LE(err, 1e-14) << "vec2 2D constant roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s0_3D_constant)
    {
        const double err = roundtrip_error_vector<2, 0, 3>(4,
                                                           [](const auto& /*cell*/)
                                                           {
                                                               return 3.14;
                                                           });
        EXPECT_LE(err, 1e-14) << "vec2 3D constant roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s0_4D_constant)
    {
        const double err = roundtrip_error_vector<2, 0, 4>(3,
                                                           [](const auto& /*cell*/)
                                                           {
                                                               return 3.14;
                                                           });
        EXPECT_LE(err, 1e-14) << "vec2 4D constant roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s0_5D_constant)
    {
        const double err = roundtrip_error_vector<2, 0, 5>(3,
                                                           [](const auto& /*cell*/)
                                                           {
                                                               return 3.14;
                                                           });
        EXPECT_LE(err, 1e-14) << "vec2 5D constant roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s0_6D_constant)
    {
        const double err = roundtrip_error_vector<2, 0, 6>(3,
                                                           [](const auto& /*cell*/)
                                                           {
                                                               return 3.14;
                                                           });
        EXPECT_LE(err, 1e-14) << "vec2 6D constant roundtrip error too large: " << err;
    }

    // --- s=1: degree-2 separable ---

    TEST(ProjectionPredictionRoundtrip, vec2_s1_1D_degree2)
    {
        const double err = roundtrip_error_vector<2, 1, 1>(4,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_1d(cell,
                                                                                  [](double x)
                                                                                  {
                                                                                      return x + (x * x);
                                                                                  });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 1D degree-2 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s1_2D_degree2)
    {
        const double err = roundtrip_error_vector<2, 1, 2>(4,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_2d(
                                                                   cell,
                                                                   [](double x)
                                                                   {
                                                                       return x + (x * x / 2.0);
                                                                   },
                                                                   [](double y)
                                                                   {
                                                                       return y + (y * y / 2.0);
                                                                   });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 2D degree-2 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s1_3D_degree2)
    {
        auto antideriv = [](double t)
        {
            return t + (t * t / 2.0);
        };
        const double err = roundtrip_error_vector<2, 1, 3>(4,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<3>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 3D degree-2 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s1_4D_degree2)
    {
        auto antideriv = [](double t)
        {
            return t + (t * t / 2.0);
        };
        const double err = roundtrip_error_vector<2, 1, 4>(3,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<4>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 4D degree-2 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s1_5D_degree2)
    {
        auto antideriv = [](double t)
        {
            return t + (t * t / 2.0);
        };
        const double err = roundtrip_error_vector<2, 1, 5>(3,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<5>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 5D degree-2 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s1_6D_degree2)
    {
        auto antideriv = [](double t)
        {
            return t + (t * t / 2.0);
        };
        const double err = roundtrip_error_vector<2, 1, 6>(3,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<6>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 6D degree-2 roundtrip error too large: " << err;
    }

    // --- s=2: degree-4 separable ---

    TEST(ProjectionPredictionRoundtrip, vec2_s2_1D_degree4)
    {
        const double err = roundtrip_error_vector<2, 2, 1>(4,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_1d(cell,
                                                                                  [](double x)
                                                                                  {
                                                                                      const double x2 = x * x;
                                                                                      const double x3 = x2 * x;
                                                                                      const double x4 = x3 * x;
                                                                                      const double x5 = x4 * x;
                                                                                      return x + (x2 / 2.0) + (x3 / 3.0) + (x4 / 4.0)
                                                                                           + (x5 / 5.0);
                                                                                  });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 1D degree-4 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s2_2D_degree4)
    {
        const double err = roundtrip_error_vector<2, 2, 2>(4,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_2d(
                                                                   cell,
                                                                   [](double x)
                                                                   {
                                                                       const double x2 = x * x;
                                                                       const double x3 = x2 * x;
                                                                       return x + (x2 / 2.0) + (x3 / 3.0);
                                                                   },
                                                                   [](double y)
                                                                   {
                                                                       const double y2 = y * y;
                                                                       const double y3 = y2 * y;
                                                                       return y + (y2 / 2.0) + (y3 / 3.0);
                                                                   });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 2D degree-4 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s2_3D_degree4)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            return t + (t2 / 2.0) + (t3 / 3.0);
        };
        const double err = roundtrip_error_vector<2, 2, 3>(4,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<3>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 3D degree-4 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s2_4D_degree4)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            return t + (t2 / 2.0) + (t3 / 3.0);
        };
        const double err = roundtrip_error_vector<2, 2, 4>(3,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<4>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 5e-13) << "vec2 4D degree-4 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s2_5D_degree4)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            return t + (t2 / 2.0) + (t3 / 3.0);
        };
        const double err = roundtrip_error_vector<2, 2, 5>(2,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<5>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 5e-12) << "vec2 5D degree-4 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s2_6D_degree4)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            return t + (t2 / 2.0) + (t3 / 3.0);
        };
        const double err = roundtrip_error_vector<2, 2, 6>(2,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<6>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 5e-11) << "vec2 6D degree-4 roundtrip error too large: " << err;
    }

    // --- s=3: degree-6 separable ---

    TEST(ProjectionPredictionRoundtrip, vec2_s3_1D_degree6)
    {
        const double err = roundtrip_error_vector<2, 3, 1>(5,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_1d(cell,
                                                                                  [](double x)
                                                                                  {
                                                                                      const double x2 = x * x;
                                                                                      const double x3 = x2 * x;
                                                                                      const double x4 = x3 * x;
                                                                                      const double x5 = x4 * x;
                                                                                      const double x6 = x5 * x;
                                                                                      const double x7 = x6 * x;
                                                                                      return x + (x2 / 2.0) + (x3 / 3.0) + (x4 / 4.0)
                                                                                           + (x5 / 5.0) + (x6 / 6.0) + (x7 / 7.0);
                                                                                  });
                                                           });
        EXPECT_LE(err, 1e-13) << "vec2 1D degree-6 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s3_2D_degree6)
    {
        const double err = roundtrip_error_vector<2, 3, 2>(5,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_2d(
                                                                   cell,
                                                                   [](double x)
                                                                   {
                                                                       const double x2 = x * x;
                                                                       const double x3 = x2 * x;
                                                                       const double x4 = x3 * x;
                                                                       return x + (x2 / 2.0) + (x3 / 3.0) + (x4 / 4.0);
                                                                   },
                                                                   [](double y)
                                                                   {
                                                                       const double y2 = y * y;
                                                                       const double y3 = y2 * y;
                                                                       const double y4 = y3 * y;
                                                                       return y + (y2 / 2.0) + (y3 / 3.0) + (y4 / 4.0);
                                                                   });
                                                           });
        EXPECT_LE(err, 5e-13) << "vec2 2D degree-6 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s3_3D_degree6)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            return t + (t2 / 2.0) + (t3 / 3.0) + (t4 / 4.0);
        };
        const double err = roundtrip_error_vector<2, 3, 3>(5,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<3>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 2e-12) << "vec2 3D degree-6 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s3_4D_degree6)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            return t + (t2 / 2.0) + (t3 / 3.0) + (t4 / 4.0);
        };
        const double err = roundtrip_error_vector<2, 3, 4>(3,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<4>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 2e-12) << "vec2 4D degree-6 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s3_5D_degree6)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            return t + (t2 / 2.0) + (t3 / 3.0) + (t4 / 4.0);
        };
        const double err = roundtrip_error_vector<2, 3, 5>(2,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<5>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 1e-11) << "vec2 5D degree-6 roundtrip error too large: " << err;
    }

    TEST(ProjectionPredictionRoundtrip, vec2_s3_6D_degree6)
    {
        auto antideriv = [](double t)
        {
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            return t + (t2 / 2.0) + (t3 / 3.0) + (t4 / 4.0);
        };
        const double err = roundtrip_error_vector<2, 3, 6>(2,
                                                           [&antideriv](const auto& cell)
                                                           {
                                                               return cell_avg_nd<6>(cell,
                                                                                     [&antideriv](std::size_t /*i*/)
                                                                                     {
                                                                                         return antideriv;
                                                                                     });
                                                           });
        EXPECT_LE(err, 5e-10) << "vec2 6D degree-6 roundtrip error too large: " << err;
    }

} // namespace samurai
