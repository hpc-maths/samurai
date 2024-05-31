// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <map>
#include <set>
#include <xtensor/xslice.hpp>

#include "ptscotch.h"

#include "mesh_interval.hpp"
#include "stencil.hpp"

using namespace xt::placeholders;

namespace samurai
{
    namespace detail
    {
        template <class Mesh, class Func>
        inline void for_each_interface_impl(const Mesh& mesh1, const Mesh& mesh2, Func&& func)
        {
            constexpr std::size_t dim        = Mesh::dim;
            using interval_t                 = typename Mesh::interval_t;
            using mesh_interval_t            = MeshInterval<dim, interval_t>;
            const std::size_t max_level_jump = 1;

            auto directions = cartesian_directions<dim>();

            for (std::size_t l_jump = 0; l_jump <= max_level_jump; ++l_jump)
            {
                if (mesh1.max_level() - mesh1.min_level() >= l_jump)
                {
                    for (std::size_t level = mesh1.min_level(); level <= mesh1.max_level() - l_jump; ++level)
                    {
                        for (std::size_t id = 0; id < directions.shape(0); ++id)
                        {
                            auto d   = xt::view(directions, id);
                            auto set = intersection(mesh1[level], translate(mesh2[level + l_jump], d)).on(level + l_jump);
                            set(
                                [&](const auto& interval, const auto& index)
                                {
                                    mesh_interval_t mesh_interval_from{level + l_jump, interval - d(0), index - xt::view(d, xt::range(1, _))};
                                    mesh_interval_t mesh_interval_to{level, interval >> l_jump, index >> l_jump};
                                    mesh_interval_from.i.step = (1 << l_jump);
                                    for (int ll = 0; ll < (1 << l_jump); ++ll)
                                    {
                                        if (mesh_interval_from.i.start == mesh_interval_from.i.end)
                                        {
                                            break;
                                        }
                                        func(mesh_interval_from, mesh_interval_to);
                                        mesh_interval_from.i.start++;
                                    }
                                });
                        }
                    }
                }
            }
        }
    }

    template <class Mesh, class Func>
    inline void for_each_inner_interface(Mesh& mesh, Func&& func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        detail::for_each_interface_impl(mesh[mesh_id_t::cells], mesh[mesh_id_t::cells], std::forward<Func>(func));
    }

    template <class Mesh, class Func>
    inline void for_each_subdomain_interface(Mesh& mesh, Func&& func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        for (auto& neigh : mesh.mpi_neighbourhood())
        {
            detail::for_each_interface_impl(mesh[mesh_id_t::cells], neigh.mesh[mesh_id_t::cells], std::forward<Func>(func));
        }
    }

    template <class Mesh, class MeshInterval>
    inline auto get_local_interval_location(const Mesh& mesh, const MeshInterval& mesh_interval)
    {
        std::size_t level  = mesh_interval.level;
        auto& i            = mesh_interval.i;
        auto& index        = mesh_interval.index;
        std::size_t offset = 0;
        for (std::size_t ll = 0; ll < level; ++ll)
        {
            offset += mesh[ll].nb_intervals(0);
        }
        return mesh[level].get_interval_location(i, index) + offset;
    }

    template <class Mesh, class MeshInterval, class LocalIndex>
    inline auto get_local_start_interval(const Mesh& mesh, const MeshInterval& mesh_interval, const LocalIndex& local_index)
    {
        return local_index[get_local_interval_location(mesh, mesh_interval)] + mesh_interval.i.start;
    }

    template <class idx_t>
    struct graph
    {
        std::vector<idx_t> global_index_offset;
        std::vector<idx_t> xadj;
        std::vector<idx_t> adjncy;
        std::vector<idx_t> adjwgt;
        std::vector<float> xyz;
    };

    template <class Mesh_t>
    auto build_graph(const Mesh_t& all_meshes)
    {
        constexpr std::size_t dim = Mesh_t::dim;
        using mesh_id_t           = typename Mesh_t::mesh_id_t;

        // graph<idx_t> g;
        graph<SCOTCH_Num> g;

        boost::mpi::communicator world;

        auto& mesh = all_meshes[mesh_id_t::cells];

        std::vector<std::size_t> global_sizes(world.size());
        mpi::all_gather(world, mesh.nb_cells(), global_sizes);

        g.global_index_offset.resize(world.size() + 1);
        for (std::size_t i = 1; i < g.global_index_offset.size(); ++i)
        {
            g.global_index_offset[i] = g.global_index_offset[i - 1] + global_sizes[i - 1];
        }

        // Construct the local index numbering of the cells
        std::vector<int> local_index(mesh.nb_intervals(0));
        std::size_t counter      = 0;
        std::size_t current_size = 0;
        for_each_interval(mesh,
                          [&](std::size_t level, const auto& i, auto& index)
                          {
                              local_index[counter++] = current_size - i.start;
                              current_size += i.size();
                          });

        // Construct the list adjacent cells for each cell
        std::map<std::size_t, std::vector<std::size_t>> adj_cells;
        std::map<std::size_t, std::vector<std::size_t>> adj_weights;
        std::size_t data_size = 0;

        // interior interface
        for_each_inner_interface(
            all_meshes,
            [&](const auto& mesh_interval_from, const auto& mesh_interval_to)
            {
                auto start_from = get_local_start_interval(mesh, mesh_interval_from, local_index) + g.global_index_offset[world.rank()];
                auto start_to   = get_local_start_interval(mesh, mesh_interval_to, local_index) + g.global_index_offset[world.rank()];

                // std::cout << "From: " << mesh_interval_from.level << " " << mesh_interval_from.i << " " << mesh_interval_from.index
                //           << std::endl;
                // std::cout << "To: " << mesh_interval_to.level << " " << mesh_interval_to.i << " " << mesh_interval_to.index << std::endl;
                auto step = mesh_interval_from.i.step;
                for (int i = 0; i < mesh_interval_to.i.size(); ++i)
                {
                    // std::cout << i << " " << step << " " << start_from + i * step << " " << start_to + i << std::endl;
                    adj_cells[start_from + i * step].push_back(start_to + i);
                    adj_weights[start_from + i * step].push_back(0.5 * (mesh_interval_from.level + mesh_interval_to.level));
                    data_size++;
                    if (mesh_interval_from.level != mesh_interval_to.level)
                    {
                        adj_cells[start_to + i].push_back(start_from + i * step);
                        adj_weights[start_to + i].push_back(0.5 * (mesh_interval_from.level + mesh_interval_to.level));
                        data_size++;
                    }
                }
                // std::cout << std::endl;
            });

        std::vector<mpi::request> req;
        std::vector<std::vector<std::size_t>> to_send(all_meshes.mpi_neighbourhood().size());
        std::set<int> recv_from;

        std::size_t i_neigh = 0;
        for (auto& neigh : all_meshes.mpi_neighbourhood())
        {
            detail::for_each_interface_impl(mesh,
                                            neigh.mesh[mesh_id_t::cells],
                                            [&](const auto&, const auto& mesh_interval)
                                            {
                                                recv_from.insert(neigh.rank);
                                                auto start = get_local_start_interval(mesh, mesh_interval, local_index)
                                                           + g.global_index_offset[world.rank()];
                                                to_send[i_neigh].push_back(static_cast<std::size_t>(start));
                                            });
            if (recv_from.find(neigh.rank) != recv_from.end())
            {
                req.push_back(world.isend(neigh.rank, neigh.rank, to_send[i_neigh++]));
            }
        }

        for (auto& neigh : all_meshes.mpi_neighbourhood())
        {
            if (recv_from.find(neigh.rank) != recv_from.end())
            {
                std::vector<std::size_t> to_recv;
                world.recv(neigh.rank, world.rank(), to_recv);

                std::size_t to_recv_counter = 0;
                detail::for_each_interface_impl(
                    neigh.mesh[mesh_id_t::cells],
                    mesh,
                    [&](const auto& mesh_interval, const auto& mesh_interval_to)
                    {
                        auto start_1 = get_local_start_interval(mesh, mesh_interval, local_index) + g.global_index_offset[world.rank()];
                        auto start_2 = to_recv[to_recv_counter++];

                        for (int i = 0, i1 = start_1, i2 = start_2; i < mesh_interval.i.size(); ++i, ++i1, ++i2)
                        {
                            adj_cells[i1].push_back(i2);
                            adj_weights[start_1 + i].push_back(0.5 * (mesh_interval.level + mesh_interval_to.level));
                            data_size += 1;
                        }
                    });
            }
        }

        mpi::wait_all(req.begin(), req.end());

        // Construct the adjacency list in CSR format
        g.xadj.resize(mesh.nb_cells() + 1);

        g.xadj[0]          = 0;
        std::size_t i_xadj = 0;
        for (auto& v : adj_cells)
        {
            g.xadj[i_xadj + 1] = g.xadj[i_xadj] + v.second.size();
            ++i_xadj;
        }

        g.adjncy.resize(g.xadj.back());
        g.adjwgt.resize(g.xadj.back());
        i_xadj = 0;
        for (auto& v : adj_cells)
        {
            std::copy(v.second.begin(), v.second.end(), g.adjncy.begin() + g.xadj[i_xadj]);
            ++i_xadj;
        }

        i_xadj = 0;
        for (auto& v : adj_weights)
        {
            std::copy(v.second.begin(), v.second.end(), g.adjwgt.begin() + g.xadj[i_xadj]);
            ++i_xadj;
        }

        std::cout << data_size << " " << g.xadj.back() << std::endl;
        // assert(data_size == g.xadj.back());
        g.xyz.reserve(mesh.nb_cells() * dim);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          auto center = cell.center();
                          std::copy(center.begin(), center.end(), std::back_inserter(g.xyz));
                      });
        return g;
    }
}
