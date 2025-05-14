#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "algorithm.hpp"
#include "algorithm/utils.hpp"
#include "mesh.hpp"
#include "mr/mesh.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif

#ifdef SAMURAI_WITH_MPI
namespace samurai
{
    enum BalanceElement_t
    {
        CELL,
        INTERVAL
    };

    /**
     * Compute the load of the current process based on intervals or cells. It uses the
     * mesh_id_t::cells to only consider leaves.
     */
    template <BalanceElement_t elem, class Mesh_t>
    static std::size_t cmptLoad(const Mesh_t& mesh)
    {
        using mesh_id_t                  = typename Mesh_t::mesh_id_t;
        const auto& current_mesh         = mesh[mesh_id_t::cells];
        std::size_t current_process_load = 0;
        // cell-based load without weight.
        samurai::for_each_interval(current_mesh,
                                   [&]([[maybe_unused]] std::size_t level, const auto& interval, [[maybe_unused]] const auto& index)
                                   {
                                       current_process_load += interval.size(); // * load_balancing_cell_weight[ level ];
                                   });
        return current_process_load;
    }

    /**
     * Compute fluxes based on load computing stategy based on graph with label
     * propagation algorithm. Return, for the current process, the flux in term of
     * load, i.e. the quantity of "load" to transfer to its neighbours. If the load
     * is negative, it means that the process (current) must send load to neighbour,
     * if positive it means that it must receive load.
     *
     * This function use 2 MPI all_gather calls.
     *
     */
    template <BalanceElement_t elem, class Mesh_t>
    std::vector<int> cmptFluxes(Mesh_t& mesh, int niterations)
    {
        using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
        boost::mpi::communicator world;
        // give access to geometricaly neighbour process rank and mesh
        std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();
        size_t n_neighbours                         = neighbourhood.size();

        // load of current process
        int my_load = static_cast<int>(cmptLoad<elem>(mesh));
        // fluxes between processes
        std::vector<int> fluxes(n_neighbours, 0);
        // load of each process (all processes not only neighbours)
        std::vector<int> loads;
        int nt = 0;
        while (nt < niterations)
        {
            boost::mpi::all_gather(world, my_load, loads);

            // compute updated my_load for current process based on its neighbourhood
            int my_load_new = my_load;
            for (std::size_t n_i = 0; n_i < n_neighbours; ++n_i)
            // get "my_load" from other processes
            {
                std::size_t neighbour_rank = static_cast<std::size_t>(neighbourhood[n_i].rank);
                int neighbour_load         = loads[neighbour_rank];
                double diff_load           = static_cast<double>(neighbour_load - my_load_new);

                // if transferLoad < 0 -> need to send data, if transferLoad > 0 need to receive data
                int transfertLoad = static_cast<int>(std::trunc(0.5 * diff_load));
                std::cout << "transfert load : " << transfertLoad << std::endl;
                fluxes[n_i] += transfertLoad;
                // my_load_new += transfertLoad;
                my_load += transfertLoad;
            }
            nt++;
        }
        return fluxes;
    }

    template <class Flavor>
    class LoadBalancer
    {
      private:

      public:

        int nloadbalancing;

        template <class Mesh_t, class Field_t>
        void update_field(Mesh_t& new_mesh, Field_t& field)
        {
            using mesh_id_t = typename Mesh_t::mesh_id_t;
            using value_t   = typename Field_t::value_type;
            boost::mpi::communicator world;

            Field_t new_field("new_f", new_mesh);
            new_field.fill(0);

            auto& old_mesh = field.mesh();
            // auto min_level = boost::mpi::all_reduce(world, mesh[mesh_id_t::cells].min_level(), boost::mpi::minimum<std::size_t>());
            // auto max_level = boost::mpi::all_reduce(world, mesh[mesh_id_t::cells].max_level(), boost::mpi::maximum<std::size_t>());
            auto min_level = old_mesh.min_level();
            auto max_level = old_mesh.max_level();

            // copy data of intervals that are didn't move
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto intersect_old_new = intersection(old_mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);
                intersect_old_new.apply_op(samurai::copy(new_field, field));
            }

            std::vector<boost::mpi::request> req, reqs;
            std::vector<std::vector<value_t>> to_send(static_cast<size_t>(world.size()));

            // TODO FIXME: this is overkill and will not scale
            // here we have to define all_* at size n_neighbours...
            std::vector<Mesh_t> all_new_meshes, all_old_meshes;
            Mesh_t recv_old_mesh, recv_new_mesh;
            for (auto& neighbour : new_mesh.mpi_neighbourhood())
            {
                std::cout << "rank ; " << neighbour.rank << std::endl;
                reqs.push_back(world.isend(neighbour.rank, 0, new_mesh));
                reqs.push_back(world.isend(neighbour.rank, 1, old_mesh));

                world.recv(neighbour.rank, 0, recv_new_mesh);
                world.recv(neighbour.rank, 1, recv_old_mesh);

                all_new_meshes.push_back(recv_new_mesh);
                all_old_meshes.push_back(recv_old_mesh);
            }
            boost::mpi::wait_all(reqs.begin(), reqs.end());

            // build payload of field that has been sent to neighbour, so compare old mesh with new neighbour mesh
            // for (auto& neighbour : new_mesh.mpi_neighbourhood())
            //            for (auto& neighbour : new_mesh.mpi_neighbourhood()){
            for (size_t ni = 0; ni < all_new_meshes.size(); ++ni)
            {
                // auto & neighbour_new_mesh = neighbour.mesh;
                auto& neighbour_new_mesh = all_new_meshes[ni];

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!old_mesh[mesh_id_t::cells][level].empty() && !neighbour_new_mesh[mesh_id_t::cells][level].empty())
                    {
                        auto intersect_old_mesh_new_neigh = intersection(old_mesh[mesh_id_t::cells][level],
                                                                         neighbour_new_mesh[mesh_id_t::cells][level]);
                        intersect_old_mesh_new_neigh(
                            [&](const auto& interval, const auto& index)
                            {
                                std::copy(field(level, interval, index).begin(),
                                          field(level, interval, index).end(),
                                          std::back_inserter(to_send[ni]));
                            });
                    }
                }

                if (to_send[ni].size() != 0)
                {
                    // neighbour_rank = neighbour.rank;
                    auto neighbour_rank = new_mesh.mpi_neighbourhood()[ni].rank;
                    req.push_back(world.isend(neighbour_rank, neighbour_rank, to_send[ni]));
                }
            }

            // build payload of field that I need to receive from neighbour, so compare NEW mesh with OLD neighbour mesh
            for (size_t ni = 0; ni < all_old_meshes.size(); ++ni)
            {
                bool isintersect = false;
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!new_mesh[mesh_id_t::cells][level].empty() && !all_old_meshes[ni][mesh_id_t::cells][level].empty())
                    {
                        std::vector<value_t> to_recv;

                        auto in_interface = intersection(all_old_meshes[ni][mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);

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
                    world.recv(new_mesh.mpi_neighbourhood()[ni].rank, world.rank(), to_recv);

                    for (std::size_t level = min_level; level <= max_level; ++level)
                    {
                        if (!new_mesh[mesh_id_t::cells][level].empty() && !all_old_meshes[ni][mesh_id_t::cells][level].empty())
                        {
                            auto in_interface = intersection(all_old_meshes[ni][mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);

                            in_interface(
                                [&](const auto& i, const auto& index)
                                {
                                    std::copy(to_recv.begin() + count,
                                              to_recv.begin() + count + static_cast<ptrdiff_t>(i.size() * field.n_comp),
                                              new_field(level, i, index).begin());
                                    count += static_cast<ptrdiff_t>(i.size() * field.n_comp);

                                    //    logs << fmt::format("Process {}, recv interval {}", world.rank(), i) << std::endl;
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
        }

        template <class Mesh_t, class Field_t, class... Fields_t>
        void update_fields(Mesh_t& new_mesh, Field_t& field, Fields_t&... kw)

        {
            update_field(new_mesh, field);
            update_fields(new_mesh, kw...);
        }

        template <class Mesh_t>
        void update_fields([[maybe_unused]] Mesh_t& new_mesh)
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
            using CellList_t  = typename Mesh_t::cl_type;
            using CellArray_t = typename Mesh_t::ca_type;

            boost::mpi::communicator world;

            CellList_t new_cl;
            // TODO why wolrd size ? scaliility ???
            std::vector<CellList_t> payload(static_cast<size_t>(world.size()));
            std::vector<size_t> payload_size(static_cast<size_t>(world.size()), 0);

            std::map<int, bool> comm;

            // build cell list for the current process && cells lists of cells for other processes
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
                            // TODO : it works ??
                            new_cl[cell.level][{cell.indices[1], cell.indices[2]}].add_point(cell.indices[0]);
                        }
                    }
                    else
                    {
                        assert(static_cast<size_t>(flags[cell]) < payload.size());

                        if (comm.find(flags[cell]) == comm.end())
                        {
                            comm[flags[cell]] = true;
                        }

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

                        payload_size[static_cast<size_t>(flags[cell])]++;
                    }
                });

            std::vector<mpi::request> req;

            // actual data echange between processes that need to exchange data
            for (int iproc = 0; iproc < world.size(); ++iproc)
            {
                if (iproc == world.rank())
                {
                    continue;
                }
                CellArray_t to_send = {payload[static_cast<size_t>(iproc)], false};
                req.push_back(world.isend(iproc, 17, to_send));
            }

            for (int iproc = 0; iproc < world.size(); ++iproc)
            {
                if (iproc == world.rank())
                {
                    continue;
                }
                CellArray_t to_rcv;
                world.recv(iproc, 17, to_rcv);

                samurai::for_each_interval(to_rcv,
                                           [&](std::size_t level, const auto& interval, const auto& index)
                                           {
                                               new_cl[level][index].add_interval(interval);
                                           });
            }
            boost::mpi::wait_all(req.begin(), req.end());
            Mesh_t new_mesh(new_cl, mesh);
            return new_mesh;
        }

        template <class Mesh_t, class Field_t, class... Fields>
        void load_balance(Mesh_t& mesh, Field_t& field, Fields&... kw)
        {
            auto flags    = static_cast<Flavor*>(this)->load_balance_impl(field.mesh());
            auto new_mesh = update_mesh(mesh, flags);

            // update each physical field on the new load balanced mesh
            update_fields(new_mesh, field, kw...);
            // swap mesh reference to new load balanced mesh. FIX: this is not clean
            field.mesh().swap(new_mesh);
            nloadbalancing += 1;
        }
    };

} // namespace samurai
#endif
