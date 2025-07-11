#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "algorithm.hpp"
#include "algorithm/utils.hpp"
#include "mesh.hpp"
#include "mr/mesh.hpp"
#include "timers.hpp"

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

    class Weight
    {
      public:
        template <class Field>
        static auto from_field(const Field& f)
        {
            auto weight = samurai::make_scalar_field<double>("weight", f.mesh());
            weight.fill(0.);
            samurai::for_each_cell(f.mesh(),
                                   [&](auto cell)
                                   {
                                       weight[cell] = f[cell];
                                   });
            return weight;
        }

        template <class Mesh>
        static auto uniform(const Mesh& mesh)
        {
            auto weight = samurai::make_scalar_field<double>("weight", mesh);
            weight.fill(1.);

            return weight;
        }

        template <BalanceElement_t elem, class Mesh_t, class Field_t>
        static double compute_load(const Mesh_t& mesh, const Field_t& weight)
        {
            using mesh_id_t                  = typename Mesh_t::mesh_id_t;
            const auto& current_mesh         = mesh[mesh_id_t::cells];
            double current_process_load = 0.;
            // cell-based load with weight.
            samurai::for_each_cell(current_mesh,
                                       [&](const auto& cell)
                                       {
                                           current_process_load += weight[cell];
                                       });
            return current_process_load;
        }
    };

    template <class Flavor>
    class LoadBalancer
    {
      public:

        int nloadbalancing;

        // Exchange only the CellArray of meshes (cells part)
        template <class Mesh_t>
        auto exchange_meshes(const Mesh_t& new_mesh, const Mesh_t& old_mesh)
        {
            samurai::times::timers.start("load_balancing_exchange_meshes");

            using CellArray_t = typename Mesh_t::ca_type;

            boost::mpi::communicator world;

            const auto& neighbours = new_mesh.mpi_neighbourhood();
            std::size_t nb_neigh   = neighbours.size();

            std::vector<CellArray_t> all_new_cells(nb_neigh);
            std::vector<CellArray_t> all_old_cells(nb_neigh);
            std::vector<mpi::request> reqs;

            // Phase 1: non-blocking receptions of CellArrays
            for (std::size_t idx = 0; idx < nb_neigh; ++idx)
            {
                const auto& nbr = neighbours[idx];
                reqs.push_back(world.irecv(nbr.rank, 0, all_new_cells[idx]));
                reqs.push_back(world.irecv(nbr.rank, 1, all_old_cells[idx]));
            }

            // Phase 2: non-blocking sends of CellArrays
            for (const auto& nbr : neighbours)
            {
                reqs.push_back(world.isend(nbr.rank, 0, new_mesh[Mesh_t::mesh_id_t::cells]));
                reqs.push_back(world.isend(nbr.rank, 1, old_mesh[Mesh_t::mesh_id_t::cells]));
            }

            // Finalize communications
            mpi::wait_all(reqs.begin(), reqs.end());

            samurai::times::timers.stop("load_balancing_exchange_meshes");

            return std::make_pair(std::move(all_new_cells), std::move(all_old_cells));
        }

        template <class Mesh_t, class Field_t>
        void update_field(Mesh_t& new_mesh,
                          Field_t& field,
                          const std::vector<typename Mesh_t::ca_type>& all_new_cells,
                          const std::vector<typename Mesh_t::ca_type>& all_old_cells)
        {
            samurai::times::timers.start("load_balancing_update_field");
            using mesh_id_t = typename Mesh_t::mesh_id_t;
            using value_t   = typename Field_t::value_type;
            boost::mpi::communicator world;

            Field_t new_field("new_f", new_mesh);
            new_field.fill(0);

            auto& old_mesh = field.mesh();
            //TODO : check if this is correct
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
            for (size_t neighbour_idx = 0; neighbour_idx < all_new_cells.size(); ++neighbour_idx)
            {
                auto& neighbour_new_cells = all_new_cells[neighbour_idx];

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!old_mesh[mesh_id_t::cells][level].empty() && !neighbour_new_cells[level].empty())
                    {
                        auto intersect_old_mesh_new_neigh = intersection(old_mesh[mesh_id_t::cells][level],
                                                                         neighbour_new_cells[level]);
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
                    auto neighbour_rank = new_mesh.mpi_neighbourhood()[neighbour_idx].rank;
                    req.push_back(world.isend(neighbour_rank, neighbour_rank, to_send[neighbour_idx]));
                }
            }

            // Build payload of field that I need to receive from neighbour, so compare NEW mesh with OLD neighbour mesh
            for (size_t neighbour_idx = 0; neighbour_idx < all_old_cells.size(); ++neighbour_idx)
            {
                bool isintersect = false;
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!new_mesh[mesh_id_t::cells][level].empty() && !all_old_cells[neighbour_idx][level].empty())
                    {
                        std::vector<value_t> to_recv;

                        auto in_interface = intersection(all_old_cells[neighbour_idx][level], new_mesh[mesh_id_t::cells][level]);

                        in_interface(
                            [&]([[maybe_unused]] const auto& i, [[maybe_unused]] const auto& index)
                            {
                                isintersect = true;
                            });

                        if (isintersect)
                        {
                            break;
                        }
                    }
                }

                if (isintersect)
                {
                    std::ptrdiff_t count = 0;
                    std::vector<value_t> to_recv;
                    world.recv(new_mesh.mpi_neighbourhood()[neighbour_idx].rank, world.rank(), to_recv);

                    for (std::size_t level = min_level; level <= max_level; ++level)
                    {
                        if (!new_mesh[mesh_id_t::cells][level].empty() && !all_old_cells[neighbour_idx][level].empty())
                        {
                            auto in_interface = intersection(all_old_cells[neighbour_idx][level], new_mesh[mesh_id_t::cells][level]);

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
            // Exchange meshes once for all fields
            auto [all_new_cells, all_old_cells] = exchange_meshes(new_mesh, field.mesh());

            // Update all fields using already exchanged CellArrays
            update_field(new_mesh, field, all_new_cells, all_old_cells);
            update_fields_impl(new_mesh, all_new_cells, all_old_cells, kw...);
        }

        template <class Mesh_t, class Field_t, class... Fields_t>
        void update_fields_impl(Mesh_t& new_mesh,
                                const std::vector<typename Mesh_t::ca_type>& all_new_cells,
                                const std::vector<typename Mesh_t::ca_type>& all_old_cells,
                                Field_t& field,
                                Fields_t&... kw)
        {
            update_field(new_mesh, field, all_new_cells, all_old_cells);
            update_fields_impl(new_mesh, all_new_cells, all_old_cells, kw...);
        }

        template <class Mesh_t>
        void update_fields_impl([[maybe_unused]] Mesh_t& new_mesh,
                                [[maybe_unused]] const std::vector<typename Mesh_t::ca_type>& all_new_cells,
                                [[maybe_unused]] const std::vector<typename Mesh_t::ca_type>& all_old_cells)
        {
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
            samurai::for_each_cell(
                mesh[Mesh_t::mesh_id_t::cells],
                [&](const auto& cell)
                {
                    if (flags[cell] == world.rank())
                    {
                        if constexpr (Mesh_t::dim == 1)
                        {
                            new_cl[cell.level][{}].add_point(cell.indices[0]);
                        }
                        if constexpr (Mesh_t::dim == 2)
                        {
                            new_cl[cell.level][{cell.indices[1]}].add_point(cell.indices[0]);
                        }
                        if constexpr (Mesh_t::dim == 3)
                        {
                            new_cl[cell.level][{cell.indices[1], cell.indices[2]}].add_point(cell.indices[0]);
                        }
                    }
                    else
                    {
                        assert(static_cast<size_t>(flags[cell]) < payload.size());

                        if constexpr (Mesh_t::dim == 1)
                        {
                            payload[static_cast<size_t>(flags[cell])][cell.level][{}].add_point(cell.indices[0]);
                        }
                        if constexpr (Mesh_t::dim == 2)
                        {
                            payload[static_cast<size_t>(flags[cell])][cell.level][{cell.indices[1]}].add_point(cell.indices[0]);
                        }
                        if constexpr (Mesh_t::dim == 3)
                        {
                            payload[static_cast<size_t>(flags[cell])][cell.level][{cell.indices[1], cell.indices[2]}].add_point(
                                cell.indices[0]);
                        }
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

        template <class Mesh_t, class Weight_t, class Field_t, class... Fields>
        void load_balance(Mesh_t& mesh, Weight_t& weight, Field_t& field, Fields&... kw)
        {
            // Early check: no load balancing with single process
            boost::mpi::communicator world;
            if (world.size() <= 1)
            {
                std::cout << "Process " << world.rank() << " : Single MPI process detected, load balancing ignored" << std::endl;
                return;
            }

            samurai::times::timers.start("load_balancing");

            // Compute flags for this single pass
            auto flags = static_cast<Flavor&>(*this).load_balance_impl(mesh, weight);

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
                using mesh_id_t = typename Mesh_t::mesh_id_t;
                double total_weight = Weight::compute_load<BalanceElement_t::CELL>(field.mesh(), weight);
                auto nb_cells = field.mesh().nb_cells(mesh_id_t::cells);
                std::cout << "Process " << world.rank() << " : " << nb_cells << " cells (total weight " << total_weight << ") after load balancing" << std::endl;
            }
        }
    };

} // namespace samurai
#endif
