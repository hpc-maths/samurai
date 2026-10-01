// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <vector>

#include "../algorithm.hpp"
#include "../array_of_interval_and_point.hpp"
#include "../stencil.hpp"
#include "neighbour_exchange.hpp"

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
                    auto domain = self(mesh.domain(level));
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
                                                         translate(self(mesh1.subdomain(level)), layer * bdry_direction));

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

    // Subdomain tag update at one level: the tags of the cells this rank owns
    // are sent to every neighbour that holds them as ghosts, which overwrites
    // (erase) or ors (default) its own tags with them.
    template <class Field>
    void update_tag_subdomains([[maybe_unused]] std::size_t level, [[maybe_unused]] Field& tag, [[maybe_unused]] bool erase = false)
    {
#ifdef SAMURAI_WITH_MPI
        using mesh_t    = typename Field::mesh_t;
        using value_t   = typename Field::value_type;
        using mesh_id_t = typename mesh_t::mesh_id_t;

        const auto& mesh          = tag.mesh();
        const auto& neighbourhood = mesh.mpi_neighbourhood();
        const auto& own           = mesh[mesh_id_t::reference][level];
        if (neighbourhood.empty() || own.empty())
        {
            return;
        }

        // out[k]: the cells of this subdomain that neighbour k holds;
        // in[k]: the cells of the subdomain of neighbour k that this rank holds.
        // out[k] here and in[.] on neighbour k are the same set (see
        // neighbour_exchange.hpp).
        const std::size_t n = neighbourhood.size();
        std::vector<detail::exchange_intervals<mesh_t>> out(n);
        std::vector<detail::exchange_intervals<mesh_t>> in(n);
        std::vector<std::size_t> send_count(n);
        std::vector<std::size_t> recv_count(n);
        for (std::size_t k = 0; k < n; ++k)
        {
            const auto& other = neighbourhood[k].mesh[mesh_id_t::reference][level];
            if (other.empty())
            {
                continue;
            }
            intersection(own, other, mesh.subdomain(level))(
                [&](const auto& i, const auto& index)
                {
                    out[k].push_back(i, index);
                });
            intersection(own, other, neighbourhood[k].mesh.subdomain(level))(
                [&](const auto& i, const auto& index)
                {
                    in[k].push_back(i, index);
                });
            send_count[k] = out[k].n_cells() * Field::n_comp;
            recv_count[k] = in[k].n_cells() * Field::n_comp;
        }

        detail::neighbour_exchange<value_t>(
            neighbourhood,
            send_count,
            recv_count,
            [&](std::size_t k, auto& buf)
            {
                detail::pack_intervals(level, tag, out[k], buf);
            },
            [&](std::size_t k, auto& it)
            {
                for (std::size_t m = 0; m < in[k].size(); ++m)
                {
                    for (auto& t : tag(level, in[k].interval(m), in[k].index(m)))
                    {
                        t = erase ? *it : static_cast<value_t>(t | *it);
                        ++it;
                    }
                }
            });
#endif
    }
}
