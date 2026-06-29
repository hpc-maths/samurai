// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * @file graph.hpp
 * @brief Distributed cell graph construction for ParMETIS and PT-Scotch.
 *
 * Builds a CSR distributed graph where each vertex is a cell and edges connect
 * face-adjacent cells (including across level jumps ≤ 1 guaranteed by
 * graduation). Vertex weights are the cell weights scaled to integers; edge
 * weights are 1 (in a Cartesian AMR grid two cells share at most one face).
 *
 * The adjacency reuses the finite-volume interface machinery: a global-index
 * field is filled on the real cells and propagated to the ghosts by the MPI /
 * periodic copy updates, then `for_each_interior_interface` enumerates every
 * interior interface (same level + level jumps, all directions, MPI neighbours
 * and periodicity) via set algebra. Each interface yields one edge.
 *
 * Communication: one all_gather (vtxdist) + the standard ghost-copy exchanges
 * of the global-index field.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "../algorithm.hpp"
#include "../algorithm/update.hpp"
#include "../field.hpp"
#include "../interface.hpp"
#include "../mesh.hpp"
#include "config.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>

namespace samurai::load_balancing
{
    inline constexpr double weight_scale_factor = 100.0;

    template <class idx_t>
    struct DistributedCellGraph
    {
        std::vector<idx_t> vtxdist;
        std::vector<idx_t> xadj;
        std::vector<idx_t> adjncy;
        std::vector<idx_t> adjwgt;
        std::vector<idx_t> vwgt;
        std::vector<float> xyz;

        idx_t nvtx_local() const noexcept
        {
            return static_cast<idx_t>(vwgt.size());
        }
    };

    /**
     * Build the distributed cell graph for the metis/scotch strategies.
     *
     * Instead of per-cell hash lookups and an O(n_neighbours * n_local) boundary
     * scan, this reuses the FV interface machinery:
     *  - a global-index field `gid` is filled on the real cells (gid = go + local
     *    order, matching the vwgt order) and propagated to the ghosts by the MPI
     *    / periodic *copy* updates. Every interface endpoint (local real cell,
     *    neighbour ghost or periodic ghost) is then a real cell of some rank, so
     *    its global ParMETIS/Scotch index is readable in O(1) via gid[cell];
     *  - `for_each_interior_interface` enumerates every interior interface
     *    (same level + level jumps l/l+1, all directions, MPI neighbours and
     *    periodicity) using intersection/translate of the per-level cell arrays.
     *    Each interface is exactly one undirected edge of the graph.
     *
     * Edge weights are 1: in a Cartesian AMR grid two cells share at most one
     * face, and a coarse cell facing several fine cells yields one distinct
     * edge (weight 1) per fine cell.
     */
    template <class idx_t, class Mesh, class Weight>
    DistributedCellGraph<idx_t> build_cell_graph(Mesh& mesh, const Weight& weight)
    {
        constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t           = typename Mesh::mesh_id_t;

        boost::mpi::communicator world;
        const int rank = world.rank();
        const int size = world.size();

        const idx_t n_local = static_cast<idx_t>(mesh.nb_cells(mesh_id_t::cells));

        // vtxdist
        DistributedCellGraph<idx_t> g;
        std::vector<idx_t> lsizes(static_cast<std::size_t>(size));
        boost::mpi::all_gather(world, n_local, lsizes);
        g.vtxdist.resize(static_cast<std::size_t>(size) + 1);
        g.vtxdist[0] = 0;
        for (int r = 0; r < size; ++r)
        {
            g.vtxdist[static_cast<std::size_t>(r) + 1] = g.vtxdist[static_cast<std::size_t>(r)] + lsizes[static_cast<std::size_t>(r)];
        }
        const idx_t go = g.vtxdist[static_cast<std::size_t>(rank)];

        // Vertex weights, coordinates and global-index field, all in the same
        // for_each_cell order so that local index == gid - go.
        double w_total = 0.;
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& c)
                      {
                          w_total += weight(c);
                      });
        const double avg_w   = (n_local > 0) ? w_total / static_cast<double>(n_local) : 1.;
        const double w_scale = (avg_w > 0.) ? weight_scale_factor / avg_w : 1.;

        g.vwgt.reserve(static_cast<std::size_t>(n_local));
        g.xyz.reserve(static_cast<std::size_t>(n_local) * dim);

        auto gid = make_scalar_field<idx_t>("lb_gid", mesh);
        gid.fill(static_cast<idx_t>(-1)); // sentinel: every interface endpoint must be overwritten

        idx_t order = 0;
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& c)
                      {
                          g.vwgt.push_back(static_cast<idx_t>(std::max(1., std::round(weight(c) * w_scale))));
                          auto ctr = c.center();
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              g.xyz.push_back(static_cast<float>(ctr[d]));
                          }
                          gid[c] = go + order;
                          ++order;
                      });

        // Propagate the global indices to the ghosts. update_mesh_neighbour is
        // required by for_each_interior_interface (it reads neigh.mesh); the
        // subdomain/periodic updates are direct copies (no prediction), which is
        // exactly what is needed since every interface endpoint is a real cell.
        mesh.update_mesh_neighbour();
        update_ghost_subdomains(gid);
        update_ghost_periodic(gid);

        // Adjacency: one edge per interior interface.
        std::vector<std::vector<idx_t>> adj(static_cast<std::size_t>(n_local));
        for_each_interior_interface(
            mesh,
            [&](const auto& interface_cells, const auto& /*comput_cells*/)
            {
                const idx_t ga = gid[interface_cells[0]];
                const idx_t gb = gid[interface_cells[1]];
                assert(ga != static_cast<idx_t>(-1) && gb != static_cast<idx_t>(-1) && "interface endpoint without a global index");
                if (ga >= go && ga < go + n_local)
                {
                    adj[static_cast<std::size_t>(ga - go)].push_back(gb);
                }
                if (gb >= go && gb < go + n_local)
                {
                    adj[static_cast<std::size_t>(gb - go)].push_back(ga);
                }
            });

        // Assemble CSR: sort + unique per vertex (weight 1 per edge).
        g.xadj.resize(static_cast<std::size_t>(n_local) + 1);
        g.xadj[0] = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_local); ++i)
        {
            auto& nbr = adj[i];
            std::sort(nbr.begin(), nbr.end());
            nbr.erase(std::unique(nbr.begin(), nbr.end()), nbr.end());
            g.xadj[i + 1] = g.xadj[i] + static_cast<idx_t>(nbr.size());
        }

        const std::size_t n_edges = static_cast<std::size_t>(g.xadj[static_cast<std::size_t>(n_local)]);
        g.adjncy.resize(n_edges);
        g.adjwgt.resize(n_edges);
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_local); ++i)
        {
            const auto off = static_cast<std::size_t>(g.xadj[i]);
            for (std::size_t j = 0; j < adj[i].size(); ++j)
            {
                g.adjncy[off + j] = adj[i][j];
                g.adjwgt[off + j] = idx_t{1};
            }
        }

        return g;
    }
}

#endif
