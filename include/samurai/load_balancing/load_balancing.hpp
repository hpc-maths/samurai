// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "../algorithm.hpp"
#include "../algorithm/utils.hpp"
#include "../mesh.hpp"
#include "../mr/mesh.hpp"
#include "../timers.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif

#ifdef SAMURAI_WITH_MPI
namespace samurai
{
    enum BalanceElement_t
    {
        CELL
    };

    class UnitWeight
    {
      public:

        template <BalanceElement_t elem, class Mesh_t>
        static double compute_load(const Mesh_t& mesh)
        {
            using mesh_id_t             = typename Mesh_t::mesh_id_t;
            const auto& current_mesh    = mesh[mesh_id_t::cells];
            double current_process_load = 0.;
            // cell-based load with weight.
            samurai::for_each_interval(current_mesh,
                                       [&](auto, const auto& interval, auto&)
                                       {
                                           current_process_load += static_cast<double>(interval.size()); // uniform weight, so just count
                                                                                                         // cells
                                       });
            return current_process_load;
        }
    };

    template <class Flavor>
    class LoadBalancer
    {
      public:

        int nloadbalancing;

        template <class Field_t>
        void update_field(typename Field_t::mesh_t& new_mesh, Field_t& field)
        {
            samurai::times::timers.start("load_balancing_update_field");
            using mesh_id_t = typename Field_t::mesh_t::mesh_id_t;
            using value_t   = typename Field_t::value_type;
            boost::mpi::communicator world;

            Field_t new_field("new_f", new_mesh);
            new_field.fill(0);

            auto& old_mesh = field.mesh();
            // TODO : check if this is correct
            auto min_level = old_mesh.min_level();
            auto max_level = old_mesh.max_level();

            // Copy data of intervals that didn't move
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto intersect_old_new = intersection(old_mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);
                intersect_old_new.apply_op(samurai::copy(new_field, field));
            }

            std::vector<boost::mpi::request> req;
            std::vector<std::vector<value_t>> to_send(static_cast<size_t>(world.size()));

            // Build payload of field that has been sent to neighbour, so compare old mesh with new neighbour mesh
            std::size_t neighbour_idx = 0;
            for (auto& neighbour : new_mesh.mpi_neighbourhood())
            {
                auto& new_cells = neighbour.mesh[mesh_id_t::cells];

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!old_mesh[mesh_id_t::cells][level].empty() && !new_cells[level].empty())
                    {
                        auto intersect_old_mesh_new_neigh = intersection(old_mesh[mesh_id_t::cells][level], new_cells[level]);
                        intersect_old_mesh_new_neigh(
                            [&](const auto& interval, const auto& index)
                            {
                                std::copy(field(level, interval, index).begin(),
                                          field(level, interval, index).end(),
                                          std::back_inserter(to_send[neighbour_idx]));
                            });
                    }
                }

                if (to_send[neighbour_idx].size() != 0)
                {
                    auto neighbour_rank = neighbour.rank;
                    req.push_back(world.isend(neighbour_rank, neighbour_rank, to_send[neighbour_idx]));
                }
                ++neighbour_idx;
            }

            neighbour_idx = 0;
            // Build payload of field that I need to receive from neighbour, so compare NEW mesh with OLD neighbour mesh
            for (auto& neighbour : old_mesh.mpi_neighbourhood())
            {
                auto& old_cells = neighbour.mesh[mesh_id_t::cells];

                bool isintersect = false;
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!new_mesh[mesh_id_t::cells][level].empty() && !old_cells[level].empty())
                    {
                        auto in_interface = intersection(old_cells[level], new_mesh[mesh_id_t::cells][level]);

                        if (!in_interface.empty())
                        {
                            isintersect = true;
                            break;
                        }
                    }
                }

                if (isintersect)
                {
                    std::ptrdiff_t count = 0;
                    std::vector<value_t> to_recv;
                    world.recv(neighbour.rank, world.rank(), to_recv);

                    for (std::size_t level = min_level; level <= max_level; ++level)
                    {
                        if (!new_mesh[mesh_id_t::cells][level].empty() && !old_cells[level].empty())
                        {
                            auto in_interface = intersection(old_cells[level], new_mesh[mesh_id_t::cells][level]);

                            in_interface(
                                [&](const auto& i, const auto& index)
                                {
                                    std::copy(to_recv.begin() + count,
                                              to_recv.begin() + count + static_cast<ptrdiff_t>(i.size() * field.n_comp),
                                              new_field(level, i, index).begin());
                                    count += static_cast<ptrdiff_t>(i.size() * field.n_comp);
                                });
                        }
                    }
                }
                ++neighbour_idx;
            }

            if (!req.empty())
            {
                mpi::wait_all(req.begin(), req.end());
            }

            std::swap(field.array(), new_field.array());
            samurai::times::timers.stop("load_balancing_update_field");
        }

        template <class Mesh_t, class Field_t, class... Fields_t>
        void update_fields(Mesh_t& new_mesh, Field_t& field, Fields_t&... kw)
        {
            update_field(new_mesh, field);
            if constexpr (sizeof...(kw) > 0)
            {
                update_fields(new_mesh, kw...);
            }
        }

      public:

        LoadBalancer()
        {
            boost::mpi::communicator world;
            nloadbalancing = 0;
        }

        template <class Mesh_t, class Field_t>
        Mesh_t update_mesh(Mesh_t& mesh, const Field_t& flags)
        {
            samurai::times::timers.start("load_balancing_mesh_update");

            using CellList_t  = typename Mesh_t::cl_type;
            using CellArray_t = typename Mesh_t::ca_type;

            boost::mpi::communicator world;

            CellList_t new_cl;
            std::vector<CellList_t> payload(static_cast<size_t>(world.size()));

            // Phase 1: build payload (cell sorting)
            samurai::times::timers.start("load_balancing_build_payload");
            samurai::for_each_cell(mesh[Mesh_t::mesh_id_t::cells],
                                   [&](const auto& cell)
                                   {
                                       auto yz_indices = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                                       if (flags[cell] == world.rank())
                                       {
                                           new_cl[cell.level][yz_indices].add_point(cell.indices[0]);
                                       }
                                       else
                                       {
                                           assert(static_cast<size_t>(flags[cell]) < payload.size());

                                           payload[static_cast<size_t>(flags[cell])][cell.level][yz_indices].add_point(cell.indices[0]);
                                       }
                                   });
            samurai::times::timers.stop("load_balancing_build_payload");

            std::vector<mpi::request> req;

            // Actual data exchange **only** with known neighbours of the mesh
            const auto& neighbours = mesh.mpi_neighbourhood();

            // Phase 2: non-blocking cell sends
            samurai::times::timers.start("load_balancing_send_cells");
            // Non-blocking send to each neighbour (possibly empty message)
            for (const auto& nbr : neighbours)
            {
                int rank = nbr.rank;
                if (rank == world.rank())
                {
                    continue;
                }

                CellArray_t to_send = {payload[static_cast<size_t>(rank)], false};
                req.push_back(world.isend(rank, 17, to_send));
            }
            samurai::times::timers.stop("load_balancing_send_cells");

            // Phase 3: cell reception
            samurai::times::timers.start("load_balancing_recv_cells");
            // Blocking reception from each neighbour
            for (const auto& nbr : neighbours)
            {
                int rank = nbr.rank;
                if (rank == world.rank())
                {
                    continue;
                }

                CellArray_t to_rcv;
                world.recv(rank, 17, to_rcv);

                samurai::for_each_interval(to_rcv,
                                           [&](std::size_t level, const auto& interval, const auto& index)
                                           {
                                               new_cl[level][index].add_interval(interval);
                                           });
            }
            samurai::times::timers.stop("load_balancing_recv_cells");

            samurai::times::timers.start("load_balancing_wait");
            boost::mpi::wait_all(req.begin(), req.end());
            samurai::times::timers.stop("load_balancing_wait");

            samurai::times::timers.start("load_balancing_construct_mesh");
            Mesh_t new_mesh(new_cl, mesh);
            samurai::times::timers.stop("load_balancing_construct_mesh");

            samurai::times::timers.stop("load_balancing_mesh_update");

            return new_mesh;
        }

        template <class Field_t, class... Fields>
        void load_balance(Field_t& field, Fields&... kw)
        {
            using Mesh_t = typename Field_t::mesh_t;
            auto& mesh   = field.mesh();
            // Early check: no load balancing with single process
            boost::mpi::communicator world;
            if (world.size() <= 1)
            {
                std::cout << "Process " << world.rank() << " : Single MPI process detected, load balancing ignored" << std::endl;
                return;
            }

            samurai::times::timers.start("load_balancing");

            // Compute flags for this single pass
            auto flags = static_cast<Flavor&>(*this).load_balance_impl(mesh);

            // Update mesh
            auto new_mesh = update_mesh(mesh, flags);

            // Update physical fields (excluding weights)
            update_fields(new_mesh, field, kw...);

            // Replace reference mesh
            mesh.swap(new_mesh);

            nloadbalancing += 1;

            samurai::times::timers.stop("load_balancing");

            // Final display of cell count after load balancing
            {
                using mesh_id_t     = typename Mesh_t::mesh_id_t;
                double total_weight = UnitWeight::compute_load<BalanceElement_t::CELL>(field.mesh());
                auto nb_cells       = field.mesh().nb_cells(mesh_id_t::cells);
                std::cout << "Process " << world.rank() << " : " << nb_cells << " cells (total weight " << total_weight
                          << ") after load balancing" << std::endl;
            }
        }
    };

} // namespace samurai
#endif
