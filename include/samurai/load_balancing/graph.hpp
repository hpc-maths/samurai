// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * @file graph.hpp
 * @brief Distributed cell graph construction for ParMETIS and PT-Scotch.
 *
 * Builds a CSR distributed graph where each vertex is a cell, edges connect
 * face-adjacent cells (including across level jumps ≤ 1 guaranteed by
 * graduation), and edge weights count the shared reference-level faces.
 * Vertex weights are the cell weights scaled to integers.
 *
 * Communication: one all_gather (vtxdist) + one point-to-point exchange per
 * MPI neighbour of boundary cell metadata.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../algorithm.hpp"
#include "../field.hpp"
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

        template <class Mesh>
        auto build_cell_index(const Mesh& mesh)
        {
            using mesh_id_t           = typename Mesh::mesh_id_t;
            using value_t             = typename Mesh::interval_t::value_t;
            constexpr std::size_t dim = Mesh::dim;
            using key_t               = CellKey<dim, value_t>;
            using map_t               = std::unordered_map<key_t, std::size_t, CellKeyHash<dim, value_t>>;

            map_t m;
            m.reserve(mesh.nb_cells(mesh_id_t::cells));
            std::size_t id = 0;
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              key_t key{cell.level, {}};
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  key.indices[d] = cell.indices[d];
                              }
                              m[key] = id++;
                          });
            return m;
        }

        template <class Mesh>
        std::size_t lookup(const std::unordered_map<CellKey<Mesh::dim, typename Mesh::interval_t::value_t>,
                                                    std::size_t,
                                                    CellKeyHash<Mesh::dim, typename Mesh::interval_t::value_t>>& idx,
                           std::size_t level,
                           const typename Mesh::cell_t::indices_t& indices)
        {
            using value_t = typename Mesh::interval_t::value_t;
            CellKey<Mesh::dim, value_t> key{level, {}};
            for (std::size_t d = 0; d < Mesh::dim; ++d)
            {
                key.indices[d] = indices[d];
            }
            auto it = idx.find(key);
            return it != idx.end() ? it->second : npos;
        }

        /** Two cells are face-adjacent iff they overlap in every dimension
         *  except exactly one, where they just touch. Level jumps are handled
         *  by projecting both cells to their finest common level. */
        template <std::size_t dim, class value_t>
        bool
        are_face_adjacent(std::size_t u_level, const std::array<value_t, dim>& u_idx, std::size_t v_level, const std::array<value_t, dim>& v_idx)
        {
            const std::size_t ref = std::max(u_level, v_level);
            const int su          = static_cast<int>(ref) - static_cast<int>(u_level);
            const int sv          = static_cast<int>(ref) - static_cast<int>(v_level);
            int touching          = 0;
            for (std::size_t d = 0; d < dim; ++d)
            {
                const long long u0 = static_cast<long long>(u_idx[d]) << su;
                const long long v0 = static_cast<long long>(v_idx[d]) << sv;
                const long long u1 = u0 + (1LL << su);
                const long long v1 = v0 + (1LL << sv);
                if (u1 == v0 || v1 == u0)
                {
                    ++touching;
                }
                else if (u1 <= v0 || v1 <= u0)
                {
                    return false;
                }
            }
            return touching == 1;
        }
    }

    template <class idx_t, class Mesh, class Weight>
    DistributedCellGraph<idx_t> build_cell_graph(Mesh& mesh, const Weight& weight)
    {
        constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t           = typename Mesh::mesh_id_t;
        using value_t             = typename Mesh::interval_t::value_t;
        using key_t               = detail::CellKey<dim, value_t>;
        using map_t               = std::unordered_map<key_t, std::size_t, detail::CellKeyHash<dim, value_t>>;

        boost::mpi::communicator world;
        const int rank = world.rank();
        const int size = world.size();

        const idx_t n_local = static_cast<idx_t>(mesh.nb_cells(mesh_id_t::cells));
        map_t id_map        = detail::build_cell_index(mesh);

        // vtxdist
        DistributedCellGraph<idx_t> g;
        std::vector<idx_t> lsizes(size);
        boost::mpi::all_gather(world, n_local, lsizes);
        g.vtxdist.resize(static_cast<std::size_t>(size) + 1);
        g.vtxdist[0] = 0;
        for (int r = 0; r < size; ++r)
        {
            g.vtxdist[static_cast<std::size_t>(r) + 1] = g.vtxdist[static_cast<std::size_t>(r)] + lsizes[static_cast<std::size_t>(r)];
        }
        const idx_t go = g.vtxdist[static_cast<std::size_t>(rank)];

        // Vertex weights & coordinates
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
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& c)
                      {
                          g.vwgt.push_back(static_cast<idx_t>(std::max(1., std::round(weight(c) * w_scale))));
                          auto ctr = c.center();
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              g.xyz.push_back(static_cast<float>(ctr[d]));
                          }
                      });

        // Adjacency lists: adj[u] = list of (global_v, edge_weight)
        std::vector<std::vector<std::pair<idx_t, idx_t>>> adj(static_cast<std::size_t>(n_local));

        // --- Internal edges: for each cell, check 2*dim directions at l-1, l, l+1 ---
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          key_t k{cell.level, {}};
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              k.indices[d] = cell.indices[d];
                          }
                          const std::size_t u = id_map.at(k);

                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              for (int sign : {-1, 1})
                              {
                                  // Same-level neighbour
                                  {
                                      typename Mesh::cell_t::indices_t ni{};
                                      for (std::size_t dd = 0; dd < dim; ++dd)
                                      {
                                          ni[dd] = cell.indices[dd];
                                      }
                                      ni[d] += static_cast<value_t>(sign);
                                      auto v = detail::lookup<Mesh>(id_map, cell.level, ni);
                                      if (v != detail::npos)
                                      {
                                          adj[u].emplace_back(go + static_cast<idx_t>(v), idx_t{1});
                                      }
                                  }
                                  // Coarser (l-1)
                                  if (cell.level > mesh.min_level())
                                  {
                                      typename Mesh::cell_t::indices_t ni{};
                                      for (std::size_t dd = 0; dd < dim; ++dd)
                                      {
                                          if (dd == d)
                                          {
                                              ni[dd] = (cell.indices[dd] + static_cast<value_t>((sign > 0) ? 1 : 0)) >> 1;
                                          }
                                          else
                                          {
                                              ni[dd] = cell.indices[dd] >> 1;
                                          }
                                      }
                                      auto v = detail::lookup<Mesh>(id_map, cell.level - 1, ni);
                                      if (v != detail::npos)
                                      {
                                          adj[u].emplace_back(go + static_cast<idx_t>(v), idx_t{1});
                                      }
                                  }
                                  // Finer (l+1): 2^(dim-1) cells on the face
                                  if (cell.level < mesh.max_level())
                                  {
                                      const value_t face   = static_cast<value_t>(2 * cell.indices[d] + ((sign > 0) ? 1 : -1));
                                      const std::size_t nf = std::size_t{1} << (dim - 1);
                                      for (std::size_t mask = 0; mask < nf; ++mask)
                                      {
                                          typename Mesh::cell_t::indices_t ni{};
                                          ni[d]           = face;
                                          std::size_t bit = 0;
                                          for (std::size_t dd = 0; dd < dim; ++dd)
                                          {
                                              if (dd == d)
                                              {
                                                  continue;
                                              }
                                              ni[dd] = static_cast<value_t>((cell.indices[dd] << 1) + ((mask >> bit) & 1));
                                              ++bit;
                                          }
                                          auto v = detail::lookup<Mesh>(id_map, cell.level + 1, ni);
                                          if (v != detail::npos)
                                          {
                                              adj[u].emplace_back(go + static_cast<idx_t>(v), idx_t{1});
                                          }
                                      }
                                  }
                              }
                          }
                      });

        // --- MPI boundary edges ---
        // Exchange (global_id, level, indices) for boundary cells with each
        // neighbour, then add edges between face-adjacent pairs.
        mesh.update_mesh_neighbour();
        auto& neighbourhood = mesh.mpi_neighbourhood();

        using bc_t = std::tuple<idx_t, std::size_t, std::array<value_t, dim>>;

        for (auto& neigh : neighbourhood)
        {
            // Collect boundary cells: cells of this rank that share a face
            // with any cell of the neighbour. We iterate over all local cells
            // and check if a face-adjacent cell exists in the neighbour's mesh
            // but not in our local hash map (meaning it belongs to the neighbour).
            // In practice, we just collect cells that have a missing neighbour
            // in a direction that is inside the neighbour's bounding box.

            std::vector<bc_t> my_bc;
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              key_t k{cell.level, {}};
                              for (std::size_t dd = 0; dd < dim; ++dd)
                              {
                                  k.indices[dd] = cell.indices[dd];
                              }
                              const std::size_t u = id_map.at(k);

                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  for (int sign : {-1, 1})
                                  {
                                      // Check same level
                                      {
                                          typename Mesh::cell_t::indices_t ni{};
                                          for (std::size_t dd = 0; dd < dim; ++dd)
                                          {
                                              ni[dd] = cell.indices[dd];
                                          }
                                          ni[d] += static_cast<value_t>(sign);
                                          auto v = detail::lookup<Mesh>(id_map, cell.level, ni);
                                          if (v == detail::npos)
                                          {
                                              // Not found locally: could be in the neighbour's mesh
                                              std::array<value_t, dim> idx{};
                                              for (std::size_t dd = 0; dd < dim; ++dd)
                                              {
                                                  idx[dd] = cell.indices[dd];
                                              }
                                              my_bc.emplace_back(go + static_cast<idx_t>(u), cell.level, idx);
                                              goto next_direction; // Only add once per cell
                                          }
                                      }
                                      // Also check level jump neighbours
                                      if (cell.level > mesh.min_level())
                                      {
                                          typename Mesh::cell_t::indices_t ni{};
                                          for (std::size_t dd = 0; dd < dim; ++dd)
                                          {
                                              if (dd == d)
                                              {
                                                  ni[dd] = (cell.indices[dd] + static_cast<value_t>((sign > 0) ? 1 : 0)) >> 1;
                                              }
                                              else
                                              {
                                                  ni[dd] = cell.indices[dd] >> 1;
                                              }
                                          }
                                          auto v = detail::lookup<Mesh>(id_map, cell.level - 1, ni);
                                          if (v == detail::npos)
                                          {
                                              std::array<value_t, dim> idx{};
                                              for (std::size_t dd = 0; dd < dim; ++dd)
                                              {
                                                  idx[dd] = cell.indices[dd];
                                              }
                                              my_bc.emplace_back(go + static_cast<idx_t>(u), cell.level, idx);
                                              goto next_direction;
                                          }
                                      }
                                      if (cell.level < mesh.max_level())
                                      {
                                          const value_t face   = static_cast<value_t>(2 * cell.indices[d] + ((sign > 0) ? 1 : -1));
                                          const std::size_t nf = std::size_t{1} << (dim - 1);
                                          for (std::size_t mask = 0; mask < nf; ++mask)
                                          {
                                              typename Mesh::cell_t::indices_t ni{};
                                              ni[d]           = face;
                                              std::size_t bit = 0;
                                              for (std::size_t dd = 0; dd < dim; ++dd)
                                              {
                                                  if (dd == d)
                                                  {
                                                      continue;
                                                  }
                                                  ni[dd] = static_cast<value_t>((cell.indices[dd] << 1) + ((mask >> bit) & 1));
                                                  ++bit;
                                              }
                                              auto v = detail::lookup<Mesh>(id_map, cell.level + 1, ni);
                                              if (v == detail::npos)
                                              {
                                                  std::array<value_t, dim> idx{};
                                                  for (std::size_t dd = 0; dd < dim; ++dd)
                                                  {
                                                      idx[dd] = cell.indices[dd];
                                                  }
                                                  my_bc.emplace_back(go + static_cast<idx_t>(u), cell.level, idx);
                                                  goto next_direction;
                                              }
                                          }
                                      }
                                  next_direction:;
                                  }
                              }
                          });

            // Deduplicate
            std::sort(my_bc.begin(), my_bc.end());
            my_bc.erase(std::unique(my_bc.begin(), my_bc.end()), my_bc.end());

            // Exchange with neighbour
            auto my_size           = static_cast<std::size_t>(my_bc.size());
            std::size_t their_size = 0;
            world.send(neigh.rank, tag_migration + 100, my_size);
            world.recv(neigh.rank, tag_migration + 100, their_size);

            std::vector<idx_t> my_ids(my_bc.size());
            std::vector<std::size_t> my_levels(my_bc.size());
            std::vector<std::array<value_t, dim>> my_coords(my_bc.size());
            for (std::size_t i = 0; i < my_bc.size(); ++i)
            {
                my_ids[i]    = std::get<0>(my_bc[i]);
                my_levels[i] = std::get<1>(my_bc[i]);
                my_coords[i] = std::get<2>(my_bc[i]);
            }

            std::vector<idx_t> their_ids(their_size);
            std::vector<std::size_t> their_levels(their_size);
            std::vector<std::array<value_t, dim>> their_coords(their_size);

            // Exchange boundary cell metadata. Avoid deadlock by ordering
            // sends: lower rank sends first, higher rank receives first.
            if (rank < neigh.rank)
            {
                world.send(neigh.rank, tag_migration + 101, my_ids);
                world.send(neigh.rank, tag_migration + 102, my_levels);
                world.send(neigh.rank, tag_migration + 103, my_coords);
                world.recv(neigh.rank, tag_migration + 101, their_ids);
                world.recv(neigh.rank, tag_migration + 102, their_levels);
                world.recv(neigh.rank, tag_migration + 103, their_coords);
            }
            else
            {
                world.recv(neigh.rank, tag_migration + 101, their_ids);
                world.recv(neigh.rank, tag_migration + 102, their_levels);
                world.recv(neigh.rank, tag_migration + 103, their_coords);
                world.send(neigh.rank, tag_migration + 101, my_ids);
                world.send(neigh.rank, tag_migration + 102, my_levels);
                world.send(neigh.rank, tag_migration + 103, my_coords);
            }

            // Add cross-rank edges using face-adjacency check
            for (std::size_t i = 0; i < my_bc.size(); ++i)
            {
                const auto u_local = static_cast<std::size_t>(std::get<0>(my_bc[i]) - go);
                const auto u_level = std::get<1>(my_bc[i]);
                const auto u_idx   = std::get<2>(my_bc[i]);

                for (std::size_t j = 0; j < their_size; ++j)
                {
                    if (detail::are_face_adjacent<dim, value_t>(u_level, u_idx, their_levels[j], their_coords[j]))
                    {
                        adj[u_local].emplace_back(their_ids[j], idx_t{1});
                    }
                }
            }
        }

        // --- Assemble CSR (sort + dedup + sum weights) ---
        g.xadj.resize(static_cast<std::size_t>(n_local) + 1);
        g.xadj[0] = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_local); ++i)
        {
            auto& edges = adj[i];
            std::sort(edges.begin(), edges.end());
            if (!edges.empty())
            {
                std::size_t write = 0;
                idx_t w           = edges[0].second;
                for (std::size_t read = 1; read < edges.size(); ++read)
                {
                    if (edges[read].first == edges[write].first)
                    {
                        w += edges[read].second;
                    }
                    else
                    {
                        edges[write].second = w;
                        ++write;
                        edges[write] = edges[read];
                        w            = edges[write].second;
                    }
                }
                edges[write].second = w;
                edges.resize(write + 1);
            }
            g.xadj[i + 1] = g.xadj[i] + static_cast<idx_t>(edges.size());
        }

        const std::size_t n_edges = static_cast<std::size_t>(g.xadj[static_cast<std::size_t>(n_local)]);
        g.adjncy.resize(n_edges);
        g.adjwgt.resize(n_edges);

        for (std::size_t i = 0; i < static_cast<std::size_t>(n_local); ++i)
        {
            idx_t off = g.xadj[i];
            for (std::size_t j = 0; j < adj[i].size(); ++j)
            {
                g.adjncy[static_cast<std::size_t>(off) + j] = adj[i][j].first;
                g.adjwgt[static_cast<std::size_t>(off) + j] = adj[i][j].second;
            }
        }

        return g;
    }
}

#endif
