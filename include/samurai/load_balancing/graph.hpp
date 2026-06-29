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
 * The adjacency reuses the finite-volume interface machinery: every interior
 * interface (same level + level jumps, all directions, MPI neighbours and
 * periodicity) is enumerated with `for_each_interior_interface` (set algebra),
 * and each interface yields one edge. Global vertex indices are resolved with a
 * map keyed on (level, indices) covering the local cells and every neighbour's
 * cells (obtained from update_mesh_neighbour, which gathers the full neighbour
 * meshes) — so no field ghost-update is needed. The distributed graph is then
 * symmetrized (PT-Scotch requires an undirected graph).
 *
 * Communication: one all_gather (vtxdist), the neighbour-mesh exchange, and one
 * all_to_all to symmetrize the boundary edges.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../algorithm.hpp"
#include "../field.hpp"
#include "../interface.hpp"
#include "../mesh.hpp"
#include "config.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

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

    namespace detail
    {
        template <std::size_t dim, class value_t>
        struct CellKey
        {
            std::size_t level{};
            std::array<value_t, dim> indices{};

            bool operator==(const CellKey& o) const
            {
                return level == o.level && indices == o.indices;
            }
        };

        template <std::size_t dim, class value_t>
        struct CellKeyHash
        {
            std::size_t operator()(const CellKey<dim, value_t>& k) const
            {
                std::size_t h = k.level;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    h ^= std::hash<value_t>{}(k.indices[d]) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
                return h;
            }
        };

        constexpr std::size_t npos = static_cast<std::size_t>(-1);

        template <std::size_t dim, class value_t, class Cell>
        CellKey<dim, value_t> cell_key(const Cell& c)
        {
            CellKey<dim, value_t> k{c.level, {}};
            for (std::size_t d = 0; d < dim; ++d)
            {
                k.indices[d] = c.indices[d];
            }
            return k;
        }
    }

    /**
     * Build the distributed cell graph for the metis/scotch strategies.
     *
     * Each interior interface (same level + level jumps l/l+1, all directions,
     * MPI neighbours and periodicity) enumerated by `for_each_interior_interface`
     * is exactly one undirected edge. Global indices are looked up in a map
     * keyed on (level, indices) holding the local cells (gid = go + local order,
     * matching the vwgt order) and every neighbour's cells (gid = neighbour
     * offset + the cell's order in the neighbour mesh, which matches the
     * neighbour's own numbering because update_mesh_neighbour copies the full
     * mesh and for_each_cell is deterministic).
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
        using value_t             = typename Mesh::interval_t::value_t;
        using key_t               = detail::CellKey<dim, value_t>;
        using map_t               = std::unordered_map<key_t, idx_t, detail::CellKeyHash<dim, value_t>>;

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

        // Global-index map (level, indices) -> global id, plus the local vertex
        // weights and coordinates (same for_each_cell order as the gid).
        map_t gid;
        gid.reserve(static_cast<std::size_t>(n_local) * 2);

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
                          gid[detail::cell_key<dim, value_t>(c)] = go + order;
                          ++order;
                      });

        // Bring in the neighbours' full meshes and number their cells with the
        // same offset/order their owner uses (deterministic for_each_cell order).
        mesh.update_mesh_neighbour();
        for (const auto& neigh : mesh.mpi_neighbourhood())
        {
            const idx_t neigh_go = g.vtxdist[static_cast<std::size_t>(neigh.rank)];
            idx_t norder         = 0;
            for_each_cell(neigh.mesh[mesh_id_t::cells],
                          [&](const auto& c)
                          {
                              gid[detail::cell_key<dim, value_t>(c)] = neigh_go + norder;
                              ++norder;
                          });
        }

        // Adjacency: one edge per interior interface. Each endpoint is a real
        // cell of some rank, hence present in the gid map.
        std::vector<std::vector<idx_t>> adj(static_cast<std::size_t>(n_local));
        for_each_interior_interface(mesh,
                                    [&](const auto& interface_cells, const auto& /*comput_cells*/)
                                    {
                                        const auto ita = gid.find(detail::cell_key<dim, value_t>(interface_cells[0]));
                                        const auto itb = gid.find(detail::cell_key<dim, value_t>(interface_cells[1]));
                                        if (ita == gid.end() || itb == gid.end())
                                        {
                                            return;
                                        }
                                        const idx_t ga = ita->second;
                                        const idx_t gb = itb->second;
                                        if (ga >= go && ga < go + n_local)
                                        {
                                            adj[static_cast<std::size_t>(ga - go)].push_back(gb);
                                        }
                                        if (gb >= go && gb < go + n_local)
                                        {
                                            adj[static_cast<std::size_t>(gb - go)].push_back(ga);
                                        }
                                    });

        // Symmetrize the distributed graph. for_each_interior_interface may
        // enumerate an MPI level-jump interface on only one of the two ranks
        // (parity of the fine interval), leaving u->v on u's owner without the
        // reverse v->u on v's owner. PT-Scotch requires a symmetric distributed
        // graph (its halo Alltoallv aborts otherwise); ParMETIS expects one too.
        // For every remote neighbour v of a local u, ask v's owner to add v->u.
        // The exchange is neighbour-to-neighbour (point-to-point, O(#neighbours)
        // not O(#ranks)): the owner of any boundary endpoint is an MPI neighbour,
        // since interfaces only connect face-adjacent cells.
        auto owner_of = [&](idx_t v)
        {
            auto it = std::upper_bound(g.vtxdist.begin(), g.vtxdist.end(), v);
            return static_cast<int>(std::distance(g.vtxdist.begin(), it)) - 1;
        };
        const auto& neighbourhood = mesh.mpi_neighbourhood();
        std::unordered_map<int, std::size_t> neigh_slot;
        for (std::size_t i = 0; i < neighbourhood.size(); ++i)
        {
            neigh_slot[neighbourhood[i].rank] = i;
        }
        std::vector<std::vector<std::pair<idx_t, idx_t>>> sym_send(neighbourhood.size());
        for (std::size_t u = 0; u < static_cast<std::size_t>(n_local); ++u)
        {
            const idx_t ug = go + static_cast<idx_t>(u);
            for (const idx_t v : adj[u])
            {
                if (v < go || v >= go + n_local) // remote neighbour
                {
                    sym_send[neigh_slot.at(owner_of(v))].emplace_back(v, ug); // owner adds v -> ug
                }
            }
        }
        std::vector<boost::mpi::request> sym_req;
        sym_req.reserve(neighbourhood.size());
        for (std::size_t i = 0; i < neighbourhood.size(); ++i)
        {
            sym_req.push_back(world.isend(neighbourhood[i].rank, tag_migration + 200, sym_send[i]));
        }
        for (const auto& neigh : neighbourhood)
        {
            std::vector<std::pair<idx_t, idx_t>> in;
            world.recv(neigh.rank, tag_migration + 200, in);
            for (const auto& [v, ug] : in)
            {
                adj[static_cast<std::size_t>(v - go)].push_back(ug);
            }
        }
        boost::mpi::wait_all(sym_req.begin(), sym_req.end());

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
