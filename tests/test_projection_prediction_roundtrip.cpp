// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>

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
        enum class FieldType : std::uint8_t
        {
            Scalar,
            Vector
        };

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
            using point_t = xt::xtensor_fixed<double, xt::xshape<dim>>;
            point_t lo;
            point_t hi;
            lo.fill(0.0);
            hi.fill(1.0);
            auto box = Box<double, dim>(lo, hi);
            return mra::make_mesh(box, mesh_cfg);
        }

        //=========================================================================
        // Helper: N-dimensional cell average for a separable polynomial.
        //
        // antiderivs(i) returns the antiderivative function for axis i.
        // cell_avg(f) = prod_i [ (A_i(c_i + h/2) - A_i(c_i - h/2)) / h ]
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
        // Helper: max absolute error over all cells and field components.
        //
        // Expected value at component c: (c+1) * cell_avg_fn(cell).
        // Works for both ScalarField (n_comp=1, is_scalar=true) and VectorField.
        //=========================================================================
        template <class Field, class CellAvgFn>
        double max_abs_error(const Field& u, std::size_t L, const CellAvgFn& cell_avg_fn)
        {
            using mesh_id_t = typename Field::mesh_t::mesh_id_t;
            double max_err  = 0.0;
            for_each_cell(u.mesh()[mesh_id_t::cells][L],
                          [&](const auto& cell)
                          {
                              const double base = cell_avg_fn(cell);
                              for (std::size_t c = 0; c < Field::n_comp; ++c)
                              {
                                  const double expected = static_cast<double>(c + 1) * base;
                                  double actual;
                                  if constexpr (Field::is_scalar)
                                  {
                                      actual = u[cell];
                                  }
                                  else
                                  {
                                      actual = u[cell][c];
                                  }
                                  max_err = std::max(max_err, std::abs(actual - expected));
                              }
                          });
            return max_err;
        }

        //=========================================================================
        // Helper: projection + prediction roundtrip.
        //
        // 1. Build two-level mesh (L-1, L).
        // 2. Fill ALL cells and ghosts at every level with exact cell averages.
        //    Ghost cells at L-1 must hold exact values so the prediction stencil
        //    has valid coarse neighbors.
        // 3. Project L -> L-1  (exact for any polynomial).
        // 4. Predict L-1 -> L  (exact for polynomials of per-axis degree <= 2s).
        // 5. Return max absolute error vs exact cell averages at level L.
        //
        // For VectorField, component c is initialised with (c+1)*cell_avg so each
        // component carries a distinct non-trivial polynomial.
        //=========================================================================
        template <FieldType ft, std::size_t n_comp, std::size_t s, std::size_t dim, class CellAvgFn>
        double roundtrip_error(std::size_t L, const CellAvgFn& cell_avg_fn)
        {
            using mesh_id_t = typename decltype(make_two_level_mesh<dim, static_cast<int>(s)>(L))::mesh_id_t;

            auto mesh = make_two_level_mesh<dim, static_cast<int>(s)>(L);

            auto u = [&]()
            {
                if constexpr (ft == FieldType::Scalar)
                {
                    return make_scalar_field<double>("u", mesh);
                }
                else
                {
                    return make_vector_field<double, n_comp>("u", mesh);
                }
            }();
            u.resize();

            for (std::size_t lvl = mesh.min_level(); lvl <= mesh.max_level(); ++lvl)
            {
                for_each_cell(mesh[mesh_id_t::reference][lvl],
                              [&](const auto& cell)
                              {
                                  const double base = cell_avg_fn(cell);
                                  if constexpr (ft == FieldType::Scalar)
                                  {
                                      u[cell] = base;
                                  }
                                  else
                                  {
                                      for (std::size_t c = 0; c < n_comp; ++c)
                                      {
                                          u[cell][c] = static_cast<double>(c + 1) * base;
                                      }
                                  }
                              });
            }

            auto proj_set = intersection(mesh[mesh_id_t::cells][L], mesh[mesh_id_t::reference][L - 1]).on(L - 1);
            proj_set.apply_op(projection(u));

            auto pred_set = intersection(mesh[mesh_id_t::cells][L], mesh[mesh_id_t::reference][L - 1]).on(L);
            pred_set.apply_op(prediction<s, false>(u));

            return max_abs_error(u, L, cell_avg_fn);
        }

        //=========================================================================
        // Dispatch helper: call roundtrip_error<ft, n_comp, s, dim> from runtime
        // (s, dim) values.  Covers s in {0,1,2,3} and dim in {1,...,6}.
        //
        // Antiderivatives of the maximal-degree separable polynomial for each s.
        //
        // The prediction operator with stencil radius s is exact for polynomials
        // whose per-axis degree is at most 2s.  For each s we test the boundary
        // case: the highest-degree polynomial the operator must reproduce exactly.
        //
        //   s=0  constant 3.14                      (no antideriv needed)
        //   s=1  u = 1 + 2t + 3t^2  (degree 2),    A(t) = t + t^2   + t^3
        //   s=2  u = 1 + t + ... + t^4 (degree 4),  A(t) = t + t^2/2 + ... + t^5/5
        //   s=3  u = 1 + t + ... + t^6 (degree 6),  A(t) = t + t^2/2 + ... + t^7/7
        //=========================================================================
        template <FieldType ft, std::size_t n_comp, std::size_t s, std::size_t dim>
        double run_case(std::size_t L)
        {
            if constexpr (s == 0)
            {
                // s=0: constant polynomial — cell average equals the constant.
                return roundtrip_error<ft, n_comp, s, dim>(L,
                                                           [](const auto& /*cell*/)
                                                           {
                                                               return 3.14;
                                                           });
            }
            else if constexpr (s == 1)
            {
                // s=1: per-axis degree 2 — A(t) = t + t^2 + t^3  (antideriv of 1+2t+3t^2)
                return roundtrip_error<ft, n_comp, s, dim>(L,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_nd<dim>(cell,
                                                                                       [](std::size_t /*i*/)
                                                                                       {
                                                                                           return [](double t)
                                                                                           {
                                                                                               const double t2 = t * t;
                                                                                               const double t3 = t2 * t;
                                                                                               return t + t2 + t3;
                                                                                           };
                                                                                       });
                                                           });
            }
            else if constexpr (s == 2)
            {
                // s=2: per-axis degree 4 — A(t) = t + t^2/2 + t^3/3 + t^4/4 + t^5/5
                return roundtrip_error<ft, n_comp, s, dim>(L,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_nd<dim>(cell,
                                                                                       [](std::size_t /*i*/)
                                                                                       {
                                                                                           return [](double t)
                                                                                           {
                                                                                               const double t2 = t * t;
                                                                                               const double t3 = t2 * t;
                                                                                               const double t4 = t3 * t;
                                                                                               const double t5 = t4 * t;
                                                                                               return t + (t2 / 2.0) + (t3 / 3.0)
                                                                                                    + (t4 / 4.0) + (t5 / 5.0);
                                                                                           };
                                                                                       });
                                                           });
            }
            else
            {
                // s=3: per-axis degree 6 — A(t) = t + t^2/2 + ... + t^7/7
                return roundtrip_error<ft, n_comp, s, dim>(L,
                                                           [](const auto& cell)
                                                           {
                                                               return cell_avg_nd<dim>(cell,
                                                                                       [](std::size_t /*i*/)
                                                                                       {
                                                                                           return [](double t)
                                                                                           {
                                                                                               const double t2 = t * t;
                                                                                               const double t3 = t2 * t;
                                                                                               const double t4 = t3 * t;
                                                                                               const double t5 = t4 * t;
                                                                                               const double t6 = t5 * t;
                                                                                               const double t7 = t6 * t;
                                                                                               return t + (t2 / 2.0) + (t3 / 3.0) + (t4 / 4.0)
                                                                                                    + (t5 / 5.0) + (t6 / 6.0) + (t7 / 7.0);
                                                                                           };
                                                                                       });
                                                           });
            }
        }

        // Dispatch on dim (1..6) for a fixed (ft, n_comp, s).
        template <FieldType ft, std::size_t n_comp, std::size_t s>
        double dispatch_dim(std::size_t dim, std::size_t L)
        {
            switch (dim)
            {
                case 1:
                    return run_case<ft, n_comp, s, 1>(L);
                case 2:
                    return run_case<ft, n_comp, s, 2>(L);
                case 3:
                    return run_case<ft, n_comp, s, 3>(L);
                case 4:
                    return run_case<ft, n_comp, s, 4>(L);
                case 5:
                    return run_case<ft, n_comp, s, 5>(L);
                case 6:
                    return run_case<ft, n_comp, s, 6>(L);
                default:
                    return -1.0;
            }
        }

        // Dispatch on s (0..3), then dim, for a fixed (ft, n_comp).
        template <FieldType ft, std::size_t n_comp>
        double dispatch_s_dim(std::size_t s, std::size_t dim, std::size_t L)
        {
            switch (s)
            {
                case 0:
                    return dispatch_dim<ft, n_comp, 0>(dim, L);
                case 1:
                    return dispatch_dim<ft, n_comp, 1>(dim, L);
                case 2:
                    return dispatch_dim<ft, n_comp, 2>(dim, L);
                case 3:
                    return dispatch_dim<ft, n_comp, 3>(dim, L);
                default:
                    return -1.0;
            }
        }

        // Top-level dispatch: also on FieldType.
        double dispatch_all(FieldType ft, std::size_t s, std::size_t dim, std::size_t L)
        {
            if (ft == FieldType::Scalar)
            {
                return dispatch_s_dim<FieldType::Scalar, 1>(s, dim, L);
            }
            return dispatch_s_dim<FieldType::Vector, 2>(s, dim, L);
        }

        //=========================================================================
        // Test parameters
        //=========================================================================
        struct RoundtripParams
        {
            FieldType ft;
            std::size_t s;
            std::size_t dim;
            std::size_t L;
            double tol;

            // Human-readable name for INSTANTIATE_TEST_SUITE_P.
            std::string name() const
            {
                const std::string ft_str = (ft == FieldType::Scalar) ? "scalar" : "vec2";
                return ft_str + "_s" + std::to_string(s) + "_" + std::to_string(dim) + "D";
            }
        };

        // Level and tolerance table.
        //
        // Level choice:
        //   - 1D–4D s=0..3: L=4 gives 16 fine cells per axis — enough for all orders.
        //   - 5D–6D s=0: L=3 (32 cells total is fine for constants).
        //   - 5D–6D s>=1: L=2 (runtime would explode at L=3: 8^5=32k, 8^6=262k cells).
        //   - 3D s=3, 4D s>=2: L=3 is sufficient and avoids excessive runtime.
        //
        // Tolerance choice (empirical, set at ~5x observed error):
        //   - Floating-point error accumulates as (2s+1)^dim interpolation terms.
        //   - Higher dim and s -> looser tolerance.
        //   - s=0 (piecewise-constant copy): machine-precision exact -> 1e-14.
        //   - s>=1 low dim: 1e-13.
        //   - s>=2 higher dim: progressively looser.
        std::vector<RoundtripParams> make_params()
        {
            // clang-format off
            // {.ft,                  .s, .dim, .L, .tol}
            std::vector<RoundtripParams> p = {
                // ---- s=0: constant -----------------------------------------------
                {.ft=FieldType::Scalar, .s=0, .dim=1, .L=4, .tol=1e-14},
                {.ft=FieldType::Scalar, .s=0, .dim=2, .L=4, .tol=1e-14},
                {.ft=FieldType::Scalar, .s=0, .dim=3, .L=4, .tol=1e-14},
                {.ft=FieldType::Scalar, .s=0, .dim=4, .L=3, .tol=1e-14},
                {.ft=FieldType::Scalar, .s=0, .dim=5, .L=3, .tol=1e-14},
                {.ft=FieldType::Scalar, .s=0, .dim=6, .L=3, .tol=1e-14},

                // ---- s=1: per-axis degree 2 ---------------------------------------
                {.ft=FieldType::Scalar, .s=1, .dim=1, .L=4, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=1, .dim=2, .L=4, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=1, .dim=3, .L=4, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=1, .dim=4, .L=3, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=1, .dim=5, .L=3, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=1, .dim=6, .L=3, .tol=5e-10},

                // ---- s=2: per-axis degree 4 ---------------------------------------
                {.ft=FieldType::Scalar, .s=2, .dim=1, .L=4, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=2, .dim=2, .L=4, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=2, .dim=3, .L=4, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=2, .dim=4, .L=3, .tol=3e-12},
                {.ft=FieldType::Scalar, .s=2, .dim=5, .L=2, .tol=3e-11},
                {.ft=FieldType::Scalar, .s=2, .dim=6, .L=2, .tol=5e-10},

                // ---- s=3: per-axis degree 6 ---------------------------------------
                {.ft=FieldType::Scalar, .s=3, .dim=1, .L=5, .tol=1e-13},
                {.ft=FieldType::Scalar, .s=3, .dim=2, .L=5, .tol=5e-13},
                {.ft=FieldType::Scalar, .s=3, .dim=3, .L=3, .tol=5e-13},
                {.ft=FieldType::Scalar, .s=3, .dim=4, .L=3, .tol=1e-11},
                {.ft=FieldType::Scalar, .s=3, .dim=5, .L=2, .tol=1e-10},
                {.ft=FieldType::Scalar, .s=3, .dim=6, .L=2, .tol=5e-09},

                // ---- vec2 s=0 ----------------------------------------------------
                {.ft=FieldType::Vector, .s=0, .dim=1, .L=4, .tol=1e-14},
                {.ft=FieldType::Vector, .s=0, .dim=2, .L=4, .tol=1e-14},
                {.ft=FieldType::Vector, .s=0, .dim=3, .L=4, .tol=1e-14},
                {.ft=FieldType::Vector, .s=0, .dim=4, .L=3, .tol=1e-14},
                {.ft=FieldType::Vector, .s=0, .dim=5, .L=3, .tol=1e-14},
                {.ft=FieldType::Vector, .s=0, .dim=6, .L=3, .tol=1e-14},

                // ---- vec2 s=1 ----------------------------------------------------
                {.ft=FieldType::Vector, .s=1, .dim=1, .L=4, .tol=1e-13},
                {.ft=FieldType::Vector, .s=1, .dim=2, .L=4, .tol=1e-13},
                {.ft=FieldType::Vector, .s=1, .dim=3, .L=4, .tol=1e-13},
                {.ft=FieldType::Vector, .s=1, .dim=4, .L=3, .tol=1e-13},
                {.ft=FieldType::Vector, .s=1, .dim=5, .L=3, .tol=1e-13},
                {.ft=FieldType::Vector, .s=1, .dim=6, .L=3, .tol=5e-10},

                // ---- vec2 s=2 ----------------------------------------------------
                {.ft=FieldType::Vector, .s=2, .dim=1, .L=4, .tol=1e-13},
                {.ft=FieldType::Vector, .s=2, .dim=2, .L=4, .tol=1e-13},
                {.ft=FieldType::Vector, .s=2, .dim=3, .L=4, .tol=5e-13},
                {.ft=FieldType::Vector, .s=2, .dim=4, .L=3, .tol=1e-11},
                {.ft=FieldType::Vector, .s=2, .dim=5, .L=2, .tol=1e-10},
                {.ft=FieldType::Vector, .s=2, .dim=6, .L=2, .tol=1e-09},

                // ---- vec2 s=3 ----------------------------------------------------
                {.ft=FieldType::Vector, .s=3, .dim=1, .L=5, .tol=1e-13},
                {.ft=FieldType::Vector, .s=3, .dim=2, .L=5, .tol=1e-12},
                {.ft=FieldType::Vector, .s=3, .dim=3, .L=3, .tol=3e-12},
                {.ft=FieldType::Vector, .s=3, .dim=4, .L=3, .tol=2e-11},
                {.ft=FieldType::Vector, .s=3, .dim=5, .L=2, .tol=2e-10},
                {.ft=FieldType::Vector, .s=3, .dim=6, .L=2, .tol=1e-08},
            };
            // clang-format on
            return p;
        }

    } // anonymous namespace

    //=========================================================================
    // Parameterized test suite
    //=========================================================================

    class RoundtripTest : public ::testing::TestWithParam<RoundtripParams>
    {
    };

    TEST_P(RoundtripTest, ProjectionPrediction)
    {
        const auto& p    = GetParam();
        const double err = dispatch_all(p.ft, p.s, p.dim, p.L);
        EXPECT_LE(err, p.tol) << p.name() << " roundtrip error too large: " << err;
    }

    INSTANTIATE_TEST_SUITE_P(ProjectionPredictionRoundtrip,
                             RoundtripTest,
                             ::testing::ValuesIn(make_params()),
                             [](const ::testing::TestParamInfo<RoundtripParams>& info)
                             {
                                 return info.param.name();
                             });

} // namespace samurai
