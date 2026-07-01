// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <vector>

#include "../algorithm.hpp"
#include "../array_of_interval_and_point.hpp"
#include "../stencil.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    template <bool to_send, class Field>
    auto outer_subdomain_corner(std::size_t level, Field& field, const typename Field::mesh_t::mpi_subdomain_t& neighbour)
    {
        using mesh_id_t  = typename Field::mesh_t::mesh_id_t;
        using lca_t      = typename Field::mesh_t::lca_type;
        using interval_t = typename Field::mesh_t::interval_t;
        using coord_t    = typename lca_t::coord_type;

        int ghost_width = field.mesh().ghost_width();

        ArrayOfIntervalAndPoint<interval_t, coord_t> interval_list;

        auto& mesh = field.mesh();
        for_each_cartesian_direction<Field::dim>(
            [&](auto bdry_direction_index, const auto& bdry_direction)
            {
                if (!mesh.is_periodic(bdry_direction_index))
                {
                    auto domain = self(mesh.domain()).on(level);
                    auto& mesh1 = to_send ? mesh : neighbour.mesh;
                    auto& mesh2 = to_send ? neighbour.mesh : mesh;

                    // The owner of an out-of-domain ghost is determined LAYER BY
                    // LAYER: the ghost at distance `layer` from the boundary
                    // belongs to the rank owning the inner cell facing it, i.e.
                    // whose subdomain translated by layer*direction covers it.
                    // Using a single translation of ghost_width for all layers
                    // (historic behaviour) designated the rank owning the cell
                    // at distance ghost_width instead: wrong as soon as the
                    // partition splits the columns adjacent to the boundary
                    // (e.g. SFC partitions), and the wrong owner then spread an
                    // unfilled value over the correctly filled one.
                    for (int layer = 1; layer <= ghost_width; ++layer)
                    {
                        // exact ghost layer `layer` in this direction
                        auto layer_band = difference(translate(domain, layer * bdry_direction),
                                                     translate(domain, (layer - 1) * bdry_direction));

                        auto owned_ghosts = intersection(mesh1[mesh_id_t::reference][level],
                                                         layer_band,
                                                         translate(self(mesh1.subdomain()).on(level), layer * bdry_direction));

                        auto neighbour_outer_corner = intersection(owned_ghosts, mesh2[mesh_id_t::reference][level]);
                        neighbour_outer_corner(
                            [&](const auto& i, const auto& index)
                            {
                                interval_list.push_back(i, index);
                            });
                    }
                }
            });

        interval_list.sort_intervals();

        lca_t lca(level);
        for (std::size_t k = 0; k < interval_list.size(); ++k)
        {
            const auto& [i, index] = interval_list[k];
            lca.add_interval_back(i, index);
        }

        return lca;
    }

    template <class Field>
    void update_tag_subdomains([[maybe_unused]] std::size_t level, [[maybe_unused]] Field& tag, [[maybe_unused]] bool erase = false)
    {
#ifdef SAMURAI_WITH_MPI
        using mesh_t    = typename Field::mesh_t;
        using value_t   = typename Field::value_type;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh = tag.mesh();
        mpi::communicator world;
        std::vector<std::vector<value_t>> to_send(mesh.mpi_neighbourhood().size());

        std::size_t i_neigh = 0;
        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                auto out_interface = intersection(mesh[mesh_id_t::reference][level],
                                                  neighbour.mesh[mesh_id_t::reference][level],
                                                  mesh.subdomain())
                                         .on(level);
                out_interface(
                    [&](const auto& i, const auto& index)
                    {
                        std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(to_send[i_neigh]));
                    });

                req.push_back(world.isend(neighbour.rank, neighbour.rank, to_send[i_neigh++]));
            }
        }

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                std::vector<value_t> to_recv;
                std::ptrdiff_t count = 0;

                world.recv(neighbour.rank, world.rank(), to_recv);

                auto in_interface = intersection(mesh[mesh_id_t::reference][level],
                                                 neighbour.mesh[mesh_id_t::reference][level],
                                                 neighbour.mesh.subdomain())
                                        .on(level);
                in_interface(
                    [&](const auto& i, const auto& index)
                    {
                        xt::xtensor<value_t, 1> neigh_tag = xt::empty_like(tag(level, i, index));
                        std::copy(to_recv.begin() + count, to_recv.begin() + count + static_cast<std::ptrdiff_t>(i.size()), neigh_tag.begin());
                        if (erase)
                        {
                            tag(level, i, index) = neigh_tag;
                        }
                        else
                        {
                            tag(level, i, index) |= neigh_tag;
                        }
                        count += static_cast<std::ptrdiff_t>(i.size());
                    });
            }
        }
        mpi::wait_all(req.begin(), req.end());

#endif
    }
}
