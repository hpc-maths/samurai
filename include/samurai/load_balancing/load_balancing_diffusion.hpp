// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <unordered_map>

#include "../field.hpp"
#include "../timers.hpp"
#include "load_balancing.hpp"

#ifdef SAMURAI_WITH_MPI
namespace samurai
{

    class DiffusionLoadBalancer : public samurai::LoadBalancer<DiffusionLoadBalancer>
    {
      private:

        template <samurai::BalanceElement_t elem, class Mesh_t, class Field_t>
        std::vector<double> compute_fluxes(Mesh_t& mesh, const Field_t& weight, int niterations)
        {
            samurai::times::timers.start("load_balancing_flux_computation");

            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
            boost::mpi::communicator world;
            std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();
            size_t n_neighbours                         = neighbourhood.size();

            // Load of current process
            double my_load = samurai::Weight::compute_load<elem>(mesh, weight);
            // Fluxes between processes
            std::vector<double> fluxes(n_neighbours, 0.);
            // Load of each process (all processes not only neighbours)
            std::vector<double> loads;
            int iteration_count = 0;
            while (iteration_count < niterations)
            {
                boost::mpi::all_gather(world, my_load, loads);

                // Compute updated my_load for current process based on its neighbourhood
                double my_load_new   = my_load;
                bool all_fluxes_zero = true;
                for (std::size_t neighbour_idx = 0; neighbour_idx < n_neighbours; ++neighbour_idx)
                {
                    std::size_t neighbour_rank = static_cast<std::size_t>(neighbourhood[neighbour_idx].rank);
                    double neighbour_load      = loads[neighbour_rank];
                    double diff_load           = neighbour_load - my_load_new;

                    // If transferLoad < 0 -> need to send data, if transferLoad > 0 need to receive data
                    // TODO : Use diffusion factor 1/(deg+1) for stability
                    double transfertLoad = 0.5 * diff_load;

                    // Accumulate total flux on current edge
                    fluxes[neighbour_idx] += transfertLoad;

                    // Mark if a non-zero transfer was performed
                    if (transfertLoad != 0)
                    {
                        all_fluxes_zero = false;
                    }

                    // Update intermediate local load before processing next neighbour
                    my_load_new += transfertLoad;
                }

                // Update reference load for next iteration
                my_load = my_load_new;

                // Check if all processes have reached convergence
                bool global_convergence = boost::mpi::all_reduce(world, all_fluxes_zero, std::logical_and<bool>());

                // If all processes have zero fluxes, state will no longer change
                if (global_convergence)
                {
                    std::cout << "Process " << world.rank() << " : Global convergence reached at iteration " << iteration_count << std::endl;
                    break;
                }

                iteration_count++;
            }

            samurai::times::timers.stop("load_balancing_flux_computation");

            return fluxes;
        }

      public:

        DiffusionLoadBalancer() = default;

        template <class Mesh_t, class Weight_t>
        auto load_balance_impl(Mesh_t& mesh, const Weight_t& weight)
        {
            using mesh_id_t = typename Mesh_t::mesh_id_t;
            boost::mpi::communicator world;

            auto flags = samurai::make_scalar_field<int>("diffusion_flag", mesh);
            flags.fill(world.rank());

            // Compute fluxes in terms of load to transfer/receive
            // Start with uniform weights to enforce row-based snapping reliably
            auto uniform_weight        = samurai::Weight::uniform(mesh);
            std::vector<double> fluxes = compute_fluxes<samurai::BalanceElement_t::CELL>(mesh, uniform_weight, 50);

            using cell_t = typename Mesh_t::cell_t;
            std::vector<cell_t> cells;
            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](auto cell)
                                   {
                                       cells.emplace_back(cell);
                                   });

            if (cells.empty())
            {
                return flags;
            }

            // Build row-based aggregation at the coarsest level to enforce a straight horizontal boundary
            static_assert(Mesh_t::dim == 2, "Row-based snapping implemented for 2D only");

            const std::size_t snap_level = mesh.min_level();

            // Aggregate cells per coarse row id
            using row_id_t = std::size_t;
            std::map<row_id_t, double> row_weight;                            // total weight per row (uniform here)
            std::unordered_map<row_id_t, std::vector<std::size_t>> row_cells; // indices of cells belonging to the row

            for (std::size_t idx = 0; idx < cells.size(); ++idx)
            {
                const auto& c         = cells[idx];
                const auto lvl        = c.level;
                const auto jy         = static_cast<row_id_t>(c.indices[1]);
                const auto d          = static_cast<unsigned>(lvl - snap_level);
                const row_id_t row_id = (d == 0) ? jy : (jy >> d);

                row_weight[row_id] += 1.0; // uniform for now
                row_cells[row_id].push_back(idx);
            }

            if (row_weight.empty())
            {
                return flags;
            }

            // Sorted unique rows (ascending: bottom -> top)
            std::vector<row_id_t> rows;
            rows.reserve(row_weight.size());
            for (const auto& kv : row_weight)
            {
                rows.push_back(kv.first);
            }

            std::size_t bottom_row = 0;
            std::size_t top_row    = rows.size() - 1;

            // Assignment of whole rows to neighbours to keep boundary strictly horizontal
            std::unordered_map<row_id_t, int> row_assignment; // row -> neighbour rank

            auto& neighbourhood = mesh.mpi_neighbourhood();

            for (std::size_t i = 0; i < neighbourhood.size(); ++i)
            {
                double flux        = fluxes[i];
                const int nbr_rank = neighbourhood[i].rank;

                if (flux < 0) // We must send rows
                {
                    double target = -flux;
                    double acc    = 0.0;

                    if (nbr_rank > world.rank())
                    {
                        // Send from the top: pick rows from the top down
                        while (acc < target && top_row >= bottom_row)
                        {
                            // Skip if already assigned (in case of prior bottom assignment)
                            while (top_row >= bottom_row && row_assignment.find(rows[top_row]) != row_assignment.end())
                            {
                                if (top_row == 0)
                                {
                                    break;
                                }
                                --top_row;
                            }
                            if (top_row < bottom_row)
                            {
                                break;
                            }

                            const row_id_t rid  = rows[top_row];
                            row_assignment[rid] = nbr_rank;
                            acc += row_weight[rid];

                            if (top_row == 0)
                            {
                                break;
                            }
                            --top_row;
                        }
                    }
                    else
                    {
                        // Send from the bottom: pick rows from the bottom up
                        while (acc < target && bottom_row <= top_row)
                        {
                            // Skip if already assigned (in case of prior top assignment)
                            while (bottom_row <= top_row && row_assignment.find(rows[bottom_row]) != row_assignment.end())
                            {
                                ++bottom_row;
                            }
                            if (bottom_row > top_row)
                            {
                                break;
                            }

                            const row_id_t rid  = rows[bottom_row];
                            row_assignment[rid] = nbr_rank;
                            acc += row_weight[rid];
                            ++bottom_row;
                        }
                    }
                }
            }

            // Apply row assignments to flags
            for (const auto& ra : row_assignment)
            {
                const row_id_t rid = ra.first;
                const int dst      = ra.second;
                const auto& idxs   = row_cells[rid];
                for (std::size_t pos : idxs)
                {
                    flags[cells[pos]] = dst;
                }
            }

            return flags;
        }
    };
}
#endif
