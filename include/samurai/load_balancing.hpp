#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "algorithm.hpp"
#include "algorithm/utils.hpp"
#include "hilbert.hpp"
#include "mesh.hpp"
#include "morton.hpp"
#include "mr/mesh.hpp"

// statistics
#ifdef WITH_STATS
#include <nlohmann/json.hpp>
#endif

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif

#ifdef SAMURAI_WITH_MPI
namespace samurai
{

    struct MPI_Load_Balance
    {
        int32_t _load;
        std::vector<int> neighbour;
        std::vector<int32_t> load;
        std::vector<int32_t> fluxes;
    };

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
        std::ofstream logs;
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

        std::ofstream logs;
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

            std::vector<boost::mpi::request> req;
            std::vector<std::vector<value_t>> to_send(static_cast<size_t>(world.size()));

            // FIXME: this is overkill and will not scale
            std::vector<Mesh_t> all_new_meshes, all_old_meshes;
            boost::mpi::all_gather(world, new_mesh, all_new_meshes);
            boost::mpi::all_gather(world, field.mesh(), all_old_meshes);

            // build payload of field that has been sent to neighbour, so compare old mesh with new neighbour mesh
            // for (auto& neighbour : new_mesh.mpi_neighbourhood())
            for (size_t ni = 0; ni < all_new_meshes.size(); ++ni)
            {
                if (static_cast<int>(ni) == world.rank())
                {
                    continue;
                }

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
                    auto neighbour_rank = static_cast<int>(ni);
                    req.push_back(world.isend(neighbour_rank, neighbour_rank, to_send[ni]));

                    //         logs << fmt::format("\t> [LoadBalancer]::update_field send data to rank # {}", neighbour_rank) << std::endl;
                }
            }

            //            logs << fmt::format("\t> [LoadBalancer]::update_field number of isend request: {}", req.size()) << std::endl;

            // build payload of field that I need to receive from neighbour, so compare NEW mesh with OLD neighbour mesh
            for (size_t ni = 0; ni < all_old_meshes.size(); ++ni)
            {
                if (static_cast<int>(ni) == world.rank())
                {
                    continue;
                }

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
                    world.recv(static_cast<int>(ni), world.rank(), to_recv);

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

        ~LoadBalancer()
        {
            logs.close();
        }

        /**
         * This function reorder cells across MPI processes based on a
         * Space Filling Curve. This is mandatory for load balancing using SFC.
         */
        template <class Mesh_t, class Field_t, class... Fields>
        void reordering(Mesh_t& mesh, Field_t& field, Fields&... kw)
        {
            // new reordered mesh on current process + MPI exchange with others
            // auto new_mesh = static_cast<Flavor*>(this)->reordering_impl( mesh );
            // logs << "\t> Computing reordering flags ... " << std::endl;
            auto flags = static_cast<Flavor*>(this)->reordering_impl(mesh);

            // logs << "\t> Update mesh based on flags ... " << std::endl;
            auto new_mesh = update_mesh(mesh, flags);

            // update each physical field on the new reordered mesh
            // SAMURAI_TRACE("[Reordering::load_balance]::Updating fields ... ");
            // logs << "\t> Update fields based on flags ... " << std::endl;
            update_fields(new_mesh, field, kw...);

            // swap mesh reference.
            // FIX: this is not clean
            // SAMURAI_TRACE("[Reordering::load_balance]::Swapping meshes ... ");
            field.mesh().swap(new_mesh);

            // discover neighbours: add new neighbours if a new interface appears or remove old neighbours
            // discover_neighbour(field.mesh());
            // discover_neighbour(field.mesh());
        }

        template <class Mesh_t, class Field_t>
        Mesh_t update_mesh(Mesh_t& mesh, const Field_t& flags)
        {
            using CellList_t  = typename Mesh_t::cl_type;
            using CellArray_t = typename Mesh_t::ca_type;

            boost::mpi::communicator world;

            CellList_t new_cl;
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

            // logs << "\t\t>[Load_balancer::update_mesh] Comm required with processes : [";
            for (const auto& it : comm)
            {
                //    logs << it.first << fmt::format(" ({} cells),", payload_size[static_cast<size_t>(it.first)]);
            }
            // logs << "]" << std::endl;

            std::vector<int> req_send(static_cast<size_t>(world.size()), 0), req_recv(static_cast<size_t>(world.size()), 0);

            // Required to know communication pattern
            for (int iproc = 0; iproc < world.size(); ++iproc)
            {
                if (iproc == world.rank())
                {
                    continue;
                }

                int reqExchg;
                comm.find(iproc) != comm.end() ? reqExchg = 1 : reqExchg = 0;

                if (payload[static_cast<size_t>(iproc)].empty())
                {
                    reqExchg = 0;
                }

                req_send[static_cast<size_t>(iproc)] = reqExchg;

                world.send(iproc, 42, reqExchg);
            }

            for (int iproc = 0; iproc < world.size(); ++iproc)
            {
                if (iproc == world.rank())
                {
                    continue;
                }
                world.recv(iproc, 42, req_recv[static_cast<size_t>(iproc)]);
            }

            for (int iproc = 0; iproc < world.size(); ++iproc)
            {
                //                logs << fmt::format("Proc # {}, req_send : {}, req_recv: {} ",
                //                                  iproc,
                //                                req_send[static_cast<size_t>(iproc)],
                //                              req_recv[static_cast<size_t>(iproc)])
                //             << std::endl;
                ;
            }

            std::vector<mpi::request> req;

            // actual data echange between processes that need to exchange data
            for (int iproc = 0; iproc < world.size(); ++iproc)
            {
                if (iproc == world.rank())
                {
                    continue;
                }

                if (req_send[static_cast<size_t>(iproc)] == 1)
                {
                    CellArray_t to_send = {payload[static_cast<size_t>(iproc)], false};

                    req.push_back(world.isend(iproc, 17, to_send));

                    // logs << fmt::format("\t> Sending to # {}", iproc) << std::endl;
                }
            }

            for (int iproc = 0; iproc < world.size(); ++iproc)
            {
                if (iproc == world.rank())
                {
                    continue;
                }

                if (req_recv[static_cast<size_t>(iproc)] == 1)
                {
                    CellArray_t to_rcv;
                    // logs << fmt::format("\t> Recving from # {}", iproc) << std::endl;
                    world.recv(iproc, 17, to_rcv);

                    samurai::for_each_interval(to_rcv,
                                               [&](std::size_t level, const auto& interval, const auto& index)
                                               {
                                                   new_cl[level][index].add_interval(interval);
                                               });
                }
            }

            boost::mpi::wait_all(req.begin(), req.end());

            Mesh_t new_mesh(new_cl, mesh);

            return new_mesh;
        }

        template <class Mesh_t, class Field_t, class... Fields>
        void load_balance(Mesh_t& mesh, Field_t& field, Fields&... kw)
        {
            if (nloadbalancing == 0)
            {
                reordering(mesh, field, kw...);
            }

            auto flags    = static_cast<Flavor*>(this)->load_balance_impl(field.mesh());
            auto new_mesh = update_mesh(mesh, flags);

            // update each physical field on the new load balanced mesh
            //            SAMURAI_TRACE("[LoadBalancer::load_balance]::Updating fields ... ");
            update_fields(new_mesh, field, kw...);
            // swap mesh reference to new load balanced mesh. FIX: this is not clean
            //            SAMURAI_TRACE("[LoadBalancer::load_balance]::Swapping meshes ... ");
            field.mesh().swap(new_mesh);
            nloadbalancing += 1;
        }
    };

    /**
     * Compute fluxes of cells between MPI processes. In -fake- MPI environment. To
     * use it in true MPI juste remove the loop over "irank", and replace irank by myrank;
     *
     */
    void compute_load_balancing_fluxes(std::vector<MPI_Load_Balance>& all)
    {
        for (size_t irank = 0; irank < all.size(); ++irank)
        {
            // number of cells
            // supposing each cell has a cost of 1. ( no level dependency )
            int32_t load = all[irank]._load;

            std::size_t n_neighbours = all[irank].neighbour.size();

            {
                std::cerr << "[compute_load_balancing_fluxes] Process # " << irank << " load : " << load << std::endl;
                std::cerr << "[compute_load_balancing_fluxes] Process # " << irank << " nneighbours : " << n_neighbours << std::endl;
                std::cerr << "[compute_load_balancing_fluxes] Process # " << irank << " neighbours : ";
                for (size_t in = 0; in < all[irank].neighbour.size(); ++in)
                {
                    std::cerr << all[irank].neighbour[in] << ", ";
                }
                std::cerr << std::endl;
            }

            // load of each process (all processes not only neighbour)
            std::vector<int64_t> loads;

            // data "load" to transfer to neighbour processes
            all[irank].fluxes.resize(n_neighbours);
            std::fill(all[irank].fluxes.begin(), all[irank].fluxes.end(), 0);

            const std::size_t n_iterations = 1;

            for (std::size_t k = 0; k < n_iterations; ++k)
            {
                // numbers of neighboors processes for each neighbour process
                std::vector<std::size_t> nb_neighbours;

                if (irank == 0)
                {
                    std::cerr << "[compute_load_balancing_fluxes] Fluxes iteration # " << k << std::endl;
                }

                // // get info from processes
                // mpi::all_gather(world, load, loads);
                // mpi::all_gather(world, m_mpi_neighbourhood.size(), nb_neighbours);

                // load of current process
                int32_t load_np1 = load;

                // compute updated load for current process based on its neighbourhood
                for (std::size_t j_rank = 0; j_rank < n_neighbours; ++j_rank)
                {
                    auto neighbour_rank = static_cast<std::size_t>(all[irank].neighbour[j_rank]);
                    auto neighbour_load = all[irank].load[j_rank];
                    auto diff_load      = neighbour_load - load;

                    std::size_t nb_neighbours_neighbour = all[neighbour_rank].neighbour.size();

                    double weight = 1. / static_cast<double>(std::max(n_neighbours, nb_neighbours_neighbour) + 1);

                    int32_t transfertLoad = static_cast<int32_t>(std::lround(weight * static_cast<double>(diff_load)));

                    all[irank].fluxes[j_rank] += transfertLoad;

                    load_np1 += transfertLoad;
                }

                // do check on load & fluxes ?

                {
                    std::cerr << "fluxes : ";
                    for (size_t in = 0; in < n_neighbours; ++in)
                    {
                        std::cerr << all[irank].fluxes[in] << ", ";
                    }
                    std::cerr << std::endl;
                }

                // load_transfer( load_fluxes );

                load = load_np1;
            }
        }
    }

} // namespace samurai
#endif
