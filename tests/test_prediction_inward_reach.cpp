// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The mesh holds what a prediction stencil reads, wherever prediction is used.
//
// Away from a boundary a stencil of radius r reads r cells each side, and the coarse levels
// carry exactly that margin. Near a boundary the stencil cannot stay centred - it shifts
// inward so that it reads only cells the domain has - and it then reaches 2r on one side.
// The margin the mesh carries has to be the larger of the two, or the shifted stencil names
// cells that are not there. That is a property of the mesh, checked here without computing
// anything: walk the positions where a detail is computed, ask where the stencil sits, and
// look up every cell it names.
//
// The second test is the other half of the same requirement, in the opposite direction: a
// cell the mesh holds but that nothing fills is worse than a cell it does not hold, because
// it is read as a value. Both must hold at once - a margin wide enough to be read, and
// narrow enough to be filled.

#include <array>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/prediction_shifts.hpp>
#include <samurai/samurai.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    namespace
    {
        // A mesh whose coarse levels are ragged, which is the situation that matters: a level
        // exists only where the projection chain needs it, so the margin around a position
        // where a detail is computed is not uniform. A ball sitting one coarse cell away from
        // the boundary, refined over a wide range of levels, produces exactly that - it is
        // the configuration of tests/test_mra.cpp, where a clamped stencil first named a cell
        // the mesh did not hold.
        template <std::size_t dim>
        auto ragged_mesh(std::size_t max_level, const std::array<bool, dim>& periodic = {})
        {
            auto cfg  = mesh_config<dim>().min_level(2).max_level(max_level).periodic(periodic);
            auto mesh = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, cfg);

            auto phi = make_scalar_field<double>("phi", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              const auto c = cell.center();
                              double r     = 0.;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r += (c[d] - 0.3) * (c[d] - 0.3);
                              }
                              phi[cell] = (std::sqrt(r) < 0.2) ? 1. : 0.;
                          });
            make_bc<Dirichlet<1>>(phi, 0.);
            auto cfg_mra = mra_config().epsilon(1e-4);
            make_MRAdapt(phi)(cfg_mra);
            return mesh;
        }

        // How many stencil cells were looked up, and how many of them the mesh did not hold.
        struct ReachSweep
        {
            std::size_t looked_up = 0;
            std::size_t missing   = 0;
        };

        // Every cell named by the stencil of one run, at `level`, in the mesh's own storage.
        template <std::size_t radius, class Mesh, class Run, class Index, class Shifts>
        void check_run(const Mesh& mesh, std::size_t level, const Run& run, const Index& index, const Shifts& shifts, ReachSweep& sweep)
        {
            static constexpr std::size_t dim  = Mesh::dim;
            static constexpr std::size_t size = 2 * radius + 1;
            using mesh_id_t                   = typename Mesh::mesh_id_t;
            using value_t                     = typename Mesh::interval_t::value_t;

            std::array<value_t, dim> start;
            for (std::size_t d = 0; d < dim; ++d)
            {
                // Where nothing fits the stencil stays centred, which is what the consumers
                // do: outside the domain it reads the outer ghosts the boundary conditions
                // write, and the mesh has those.
                const int shift = shifts.fits ? shifts.shift[d] : 0;
                start[d]        = static_cast<value_t>(-static_cast<int>(radius) + shift);
            }

            for (auto x = run.start; x < run.end; ++x)
            {
                static_nested_loop<dim, 0, size>(
                    [&](const auto& k)
                    {
                        xt::xtensor_fixed<value_t, xt::xshape<dim>> c;
                        c[0] = x + start[0] + static_cast<value_t>(k[0]);
                        for (std::size_t d = 1; d < dim; ++d)
                        {
                            c[d] = index[d - 1] + start[d] + static_cast<value_t>(k[d]);
                        }

                        ++sweep.looked_up;
                        if (find(mesh[mesh_id_t::reference][level], c) < 0)
                        {
                            if (sweep.missing < 5)
                            {
                                std::string at = "(";
                                for (std::size_t d = 0; d < dim; ++d)
                                {
                                    at += (d == 0 ? "" : ",") + std::to_string(c[d]);
                                }
                                ADD_FAILURE() << "level " << level << ": the cell at x = " << x << " names " << at << ")"
                                              << " which the mesh does not hold; shifts fit=" << shifts.fits
                                              << " shift0=" << shifts.shift[0] << " shift1=" << shifts.shift[1];
                            }
                            ++sweep.missing;
                        }
                    });
            }
        }

        // The positions where the detail is computed are the ones adapt visits: every cell of
        // all_cells lying below a leaf. Same set, so the same stencils.
        template <class Mesh>
        ReachSweep sweep_detail_positions(const Mesh& mesh)
        {
            static constexpr std::size_t radius = Mesh::config_t::prediction_stencil_radius;
            using mesh_id_t                     = typename Mesh::mesh_id_t;

            using value_t = typename Mesh::interval_t::value_t;

            // The wrap the periodic ghost exchange copies by, seen at this level, and 0 where
            // the direction is not periodic. Spelled out here rather than shared: a test that
            // computed it the same way as the code under test would not be checking it.
            const auto period_at = [&](std::size_t level)
            {
                std::array<value_t, Mesh::dim> period{};
                const auto minmax  = mesh.domain().minmax_indices();
                const auto delta_l = mesh.domain().level() - level;
                for (std::size_t d = 0; d < Mesh::dim; ++d)
                {
                    period[d] = mesh.is_periodic(d) ? static_cast<value_t>((minmax[d].second - minmax[d].first) >> delta_l) : 0;
                }
                return period;
            };

            ReachSweep sweep;
            for (std::size_t level = mesh.min_level() > 0 ? mesh.min_level() - 1 : 0; level < mesh.max_level(); ++level)
            {
                const auto period = period_at(level);

                auto ghosts_below_cells = intersection(mesh[mesh_id_t::all_cells][level],
                                                       union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                              .on(level);
                ghosts_below_cells(
                    [&](const auto& i, const auto& index)
                    {
                        for_each_prediction_shift_run<radius>(mesh.domain(level),
                                                              period,
                                                              i,
                                                              index,
                                                              [&](const auto& run, const auto& shifts)
                                                              {
                                                                  check_run<radius>(mesh, level, run, index, shifts, sweep);
                                                              });
                    });
            }
            return sweep;
        }
    }

    TEST(prediction_inward_reach, the_mesh_holds_what_a_clamped_stencil_reads_2d)
    {
        const auto mesh = ragged_mesh<2>(10);

        const auto sweep = sweep_detail_positions(mesh);
        EXPECT_GT(sweep.looked_up, 0u);
        EXPECT_EQ(sweep.missing, 0u);
    }

    TEST(prediction_inward_reach, the_mesh_holds_what_a_clamped_stencil_reads_1d)
    {
        const auto mesh = ragged_mesh<1>(8);

        const auto sweep = sweep_detail_positions(mesh);
        EXPECT_GT(sweep.looked_up, 0u);
        EXPECT_EQ(sweep.missing, 0u);
    }

    TEST(prediction_inward_reach, the_mesh_holds_what_a_clamped_stencil_reads_3d)
    {
        const auto mesh = ragged_mesh<3>(7);

        const auto sweep = sweep_detail_positions(mesh);
        EXPECT_GT(sweep.looked_up, 0u);
        EXPECT_EQ(sweep.missing, 0u);
    }

    TEST(prediction_inward_reach, the_mesh_holds_what_a_clamped_stencil_reads_when_periodic)
    {
        // A periodic direction has no boundary, so nothing is clamped there; the other one
        // still is, and the two must coexist on one mesh.
        const auto mesh = ragged_mesh<2>(10, {true, false});

        const auto sweep = sweep_detail_positions(mesh);
        EXPECT_GT(sweep.looked_up, 0u);
        EXPECT_EQ(sweep.missing, 0u);
    }

    TEST(prediction_inward_reach, the_mesh_holds_what_a_clamped_stencil_reads_while_adapting)
    {
        // The invariant has to hold on every mesh adaptation builds, not only on the one it
        // settles on: the first cell a clamped stencil named and the mesh did not hold showed
        // up on an intermediate mesh, inside adapt. A moving front is what produces those.
        constexpr std::size_t dim = 2;

        auto cfg  = mesh_config<dim>().min_level(2).max_level(8);
        auto mesh = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, cfg);

        auto phi = make_scalar_field<double>("phi", mesh);
        make_bc<Dirichlet<1>>(phi, 0.);
        auto adapt = make_MRAdapt(phi);

        std::size_t looked_up = 0;
        for (std::size_t step = 0; step < 6; ++step)
        {
            const double centre = 0.2 + 0.1 * static_cast<double>(step);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              const auto c = cell.center();
                              double r     = 0.;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r += (c[d] - centre) * (c[d] - centre);
                              }
                              phi[cell] = (std::sqrt(r) < 0.2) ? 1. : 0.;
                          });

            auto cfg_mra = mra_config().epsilon(1e-4);
            adapt(cfg_mra);

            const auto sweep = sweep_detail_positions(mesh);
            EXPECT_EQ(sweep.missing, 0u) << "at step " << step;
            looked_up += sweep.looked_up;
        }
        EXPECT_GT(looked_up, 0u);
    }

    TEST(prediction_inward_reach, every_cell_the_mesh_holds_is_filled)
    {
        // The margin must not outrun what projection, prediction and the boundary conditions
        // between them write: a cell that is held but never written is read as a value.
        constexpr std::size_t dim = 2;
        using mesh_id_t           = MRMeshId;

        auto mesh = ragged_mesh<dim>(10);

        auto u = make_scalar_field<double>("u", mesh);
        u.fill(std::nan(""));
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          const auto c = cell.center();
                          u[cell]      = c[0] + 2. * c[1];
                      });
        make_bc<Dirichlet<1>>(u, 0.);
        update_ghost_mr(u);

        std::size_t unfilled = 0;
        std::size_t total    = 0;
        for_each_cell(mesh[mesh_id_t::reference],
                      [&](const auto& cell)
                      {
                          ++total;
                          if (std::isnan(u[cell]))
                          {
                              if (unfilled < 5)
                              {
                                  const auto& idx = cell.indices;
                                  ADD_FAILURE() << "level " << cell.level << ": the cell at (" << idx[0] << "," << idx[1] << ") is held by"
                                                << " the mesh and was never written";
                              }
                              ++unfilled;
                          }
                      });

        EXPECT_GT(total, 0u);
        EXPECT_EQ(unfilled, 0u);
    }
}
