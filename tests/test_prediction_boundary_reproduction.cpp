// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// A polynomial the prediction operator reproduces has no detail - at the boundary too.
//
// This is the invariant the inward-shifted stencils exist for, read through the library
// rather than off the coefficients. Prediction reproduces every monomial of degree <= 2r
// *per variable*, so a field holding the cell averages of such a polynomial is predicted
// exactly and every detail is zero. That used to fail at the boundary for one reason: a
// cell touching it predicted from ghosts written by the boundary conditions, which
// reproduce nothing of the sort, so the wavelet reported a detail where the solution is a
// polynomial. Adaptation then refined there, and the threshold it compared against was not
// measuring the error it was supposed to measure.
//
// Both sides are pinned, as elsewhere in this suite: exact at degree 2r, and NOT exact one
// degree above. Exactness alone would also hold for an operator of higher order than the
// one claimed, so it does not pin an order by itself.
//
// The field is filled with **cell averages**, computed by samurai's own GaussLegendre at a
// degree well above the polynomials used, because a finite-volume field stores averages and
// the prediction coefficients are solved from the cell-average moment conditions. Filling
// with point values would test a different operator.

#include <cmath>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/operators.hpp>
#include <samurai/numeric/gauss_legendre.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    namespace
    {
        template <std::size_t dim, class TInterval, class F>
        double cell_average(const Cell<dim, TInterval>& cell, F&& f)
        {
            static GaussLegendre<10> gl;
            return gl.template quadrature<1>(cell, f) / std::pow(cell.length, static_cast<double>(dim));
        }

        // An adapted mesh whose refinement reaches the boundary, so that details are
        // computed at cells whose stencil would otherwise read outside the domain. A sharp
        // ball at the origin corner refines the two '-' boundaries there.
        template <std::size_t dim>
        auto adapted_mesh()
        {
            auto cfg  = mesh_config<dim>().min_level(2).max_level(5);
            auto mesh = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, cfg);

            auto phi = make_scalar_field<double>("phi", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              const auto c = cell.center();
                              double r     = 0.;
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

        // The largest detail over the whole mesh, and how many were looked at, for a field
        // holding the cell averages of f. Returned rather than asserted so that a caller can
        // demand exactness or demand its absence.
        struct DetailSweep
        {
            std::size_t nb = 0;
            double max_abs = 0.;
        };

        template <class Mesh, class F>
        DetailSweep sweep_details(Mesh& mesh, F&& f)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            auto u = make_scalar_field<double>("u", mesh);
            u.fill(0.);
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](const auto& cell)
                          {
                              u[cell] = cell_average(cell, f);
                          });

            // The boundary conditions still fill the outer ghosts; the point of the test is
            // that prediction no longer reads them.
            make_bc<Dirichlet<1>>(u, 0.);
            update_ghost_mr(u);

            auto detail = make_scalar_field<double>("detail", mesh);
            detail.fill(0.);

            DetailSweep sweep;
            for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
            {
                auto ghosts_below_cells = intersection(mesh[mesh_id_t::all_cells][level],
                                                       union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                              .on(level);
                ghosts_below_cells.apply_op(compute_detail(detail, u));

                for_each_cell(mesh[mesh_id_t::cells][level + 1],
                              [&](const auto& cell)
                              {
                                  ++sweep.nb;
                                  sweep.max_abs = std::max(sweep.max_abs, std::abs(detail[cell]));
                              });
            }
            return sweep;
        }
    }

    // The detail a centred stencil would give, computed here rather than by the library.
    template <std::size_t radius, class Field>
    double centred_detail(const Field& u, std::size_t level, int i, int j, std::size_t parity_i, std::size_t parity_j)
    {
        const auto& cx = prediction_coefficients<radius>(parity_i, 0).c;
        const auto& cy = prediction_coefficients<radius>(parity_j, 0).c;

        const int child_i = 2 * i + static_cast<int>(parity_i);
        const int child_j = 2 * j + static_cast<int>(parity_j);

        // The kernel starts from the child value and subtracts one term per stencil point, kj
        // outermost, each coefficient formed as (cx * cy) * src. Mirrored exactly: a different
        // summation order would give a different double, and the point here is bit-identity.
        double d = u(level + 1, typename Field::interval_t{child_i, child_i + 1}, child_j)(0);
        for (std::size_t kj = 0; kj < 2 * radius + 1; ++kj)
        {
            for (std::size_t ki = 0; ki < 2 * radius + 1; ++ki)
            {
                const int si = i + static_cast<int>(ki) - static_cast<int>(radius);
                const int sj = j + static_cast<int>(kj) - static_cast<int>(radius);
                d -= (cx[ki] * cy[kj]) * u(level, typename Field::interval_t{si, si + 1}, sj)(0);
            }
        }
        return d;
    }

    TEST(prediction_boundary_reproduction, the_interior_is_bit_identical)
    {
        // Acceptance bar item 1, in the library rather than as an argument about the code: away
        // from every boundary the stencil is centred, and the detail must be *exactly* what the
        // centred formula gives - not close to it. The field is deliberately not a polynomial,
        // so the details are large and an equality is worth something.
        //
        // "Away from every boundary" is decided from the domain's own extent here, not from the
        // query, so this test does not inherit the query's opinion of where a boundary is.
        constexpr std::size_t dim    = 2;
        constexpr std::size_t radius = mesh_config<dim>::prediction_stencil_radius;
        using mesh_id_t              = MRMeshId;

        auto mesh = adapted_mesh<dim>();

        auto u = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh[mesh_id_t::reference],
                      [&](const auto& cell)
                      {
                          const auto c = cell.center();
                          u[cell]      = std::exp(3. * c[0]) * std::cos(7. * c[1]);
                      });
        make_bc<Dirichlet<1>>(u, 0.);
        update_ghost_mr(u);

        auto detail = make_scalar_field<double>("detail", mesh);
        detail.fill(0.);

        std::size_t compared = 0;
        for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
        {
            auto ghosts_below_cells = intersection(mesh[mesh_id_t::all_cells][level],
                                                   union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                          .on(level);
            ghosts_below_cells.apply_op(compute_detail(detail, u));

            const auto minmax  = mesh.domain().minmax_indices();
            const auto delta_l = mesh.domain().level() - level;
            const int reach    = 2 * static_cast<int>(radius);

            ghosts_below_cells(
                [&](const auto& itv, const auto& index)
                {
                    const int j = index[0];
                    if (j - static_cast<int>(minmax[1].first >> delta_l) < reach
                        || static_cast<int>(minmax[1].second >> delta_l) - 1 - j < reach)
                    {
                        return; // the transverse row is within reach of a boundary
                    }
                    for (int i = itv.start; i < itv.end; ++i)
                    {
                        if (i - static_cast<int>(minmax[0].first >> delta_l) < reach
                            || static_cast<int>(minmax[0].second >> delta_l) - 1 - i < reach)
                        {
                            continue;
                        }
                        for (std::size_t pj = 0; pj < 2; ++pj)
                        {
                            for (std::size_t pi = 0; pi < 2; ++pi)
                            {
                                const double expected = centred_detail<radius>(u, level, i, j, pi, pj);
                                const int ci          = 2 * i + static_cast<int>(pi);
                                const int cj          = 2 * j + static_cast<int>(pj);
                                const double got      = detail(level + 1, typename decltype(detail)::interval_t{ci, ci + 1}, cj)(0);
                                EXPECT_EQ(got, expected) << "at level " << level << " cell (" << ci << "," << cj << ")";
                                ++compared;
                            }
                        }
                    }
                });
        }

        EXPECT_GT(compared, 0u);
    }

    TEST(prediction_boundary_reproduction, a_reproduced_polynomial_has_no_detail_1d)
    {
        constexpr double two_r = 2. * static_cast<double>(mesh_config<1>::prediction_stencil_radius);

        auto mesh = adapted_mesh<1>();

        // Degree 2r: reproduced, so no detail anywhere.
        const auto reproduced = sweep_details(mesh,
                                              [](const auto& x)
                                              {
                                                  return std::pow(x(0), two_r);
                                              });
        EXPECT_GT(reproduced.nb, 0u);
        EXPECT_LT(reproduced.max_abs, 1e-12);

        // One degree above: not reproduced, so the detail must be there. Without this the
        // test above would also pass for an operator of higher order than the one claimed.
        const auto beyond = sweep_details(mesh,
                                          [](const auto& x)
                                          {
                                              return std::pow(x(0), two_r + 1.);
                                          });
        EXPECT_GT(beyond.max_abs, 1e-8);
    }

    TEST(prediction_boundary_reproduction, a_reproduced_polynomial_has_no_detail_2d)
    {
        constexpr double two_r = 2. * static_cast<double>(mesh_config<2>::prediction_stencil_radius);

        auto mesh = adapted_mesh<2>();

        // Q_{2r}, not P_{2r}: the operator is a tensor product, so the degree that counts is
        // the degree in each variable separately. x^2 y^2 is reproduced at r = 1; x^3 is not.
        const auto reproduced = sweep_details(mesh,
                                              [](const auto& x)
                                              {
                                                  return std::pow(x(0), two_r) * std::pow(x(1), two_r);
                                              });
        EXPECT_GT(reproduced.nb, 0u);
        EXPECT_LT(reproduced.max_abs, 1e-12);

        const auto beyond = sweep_details(mesh,
                                          [](const auto& x)
                                          {
                                              return std::pow(x(0), two_r + 1.);
                                          });
        EXPECT_GT(beyond.max_abs, 1e-8);
    }
}
