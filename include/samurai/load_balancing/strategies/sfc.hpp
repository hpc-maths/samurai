// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * Weighted space-filling curve partitioning (p4est-like).
 *
 * Algorithm:
 *  1. every cell gets the SFC key of its position normalized to the global
 *     max level (`indices << (max_level - level)`), after a global shift
 *     making all coordinates non-negative (curves require >= 0);
 *  2. local cells are sorted by key: the mesh becomes a set of P chunks of
 *     one global 1D sequence;
 *  3. the curve is cut into P segments of equal weight by locating the P-1 cut
 *     keys s_r = smallest key c such that the *global* weight of the cells with
 *     key < c reaches r * (W_total / P). They are found with a vectorised
 *     binary search over the key space (one all_reduce per bisection step), so
 *     no gather of keys or cells is needed;
 *  4. a cell of key k goes to rank #{ r : s_r <= k }, i.e. floor(c / (W_total /
 *     P)) where c is its global exclusive-prefix weight along the curve.
 *
 * Guarantees:
 *  - weighted balance up to the heaviest single cell per rank;
 *  - partitions are contiguous segments of the curve: compact for Hilbert
 *    (continuous curve), good for Morton (jumps at power-of-two boundaries);
 *  - deterministic: keys are unique (cells do not overlap), so the global
 *    order is total and two consecutive calls produce the same partition
 *    (second call migrates nothing);
 *  - correct from *any* initial decomposition: the cut keys depend only on the
 *    global cumulative weight, never on which rank currently owns a cell.
 *
 * Why a global search and not an MPI scan: a scan over the rank index yields
 * the weight of the lower-index ranks, which equals the curve prefix only if
 * each rank already owns a curve-contiguous segment. The default decomposition
 * (mesh.hpp::partition_mesh) splits the intervals row-major, so the ranks'
 * key chunks interleave on the curve; a scan-based offset would then mis-assign
 * cells and fracture every rank into disconnected islands -- even for Hilbert.
 *
 * Communication: 3 all_reduce for the setup (2 for the coordinate shift bound,
 * 1 for the total weight + 1 for the key bound) plus <= 64 all_reduce of P-1
 * doubles for the cut search. No data gather.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include "../../field.hpp"
#include "../sfc/hilbert.hpp"
#include "../sfc/morton.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>

namespace samurai::load_balancing
{
    template <class Curve>
    class SFC
    {
      public:

        SFC() = default;

        explicit SFC(Curve curve)
            : m_curve(std::move(curve))
        {
        }

        std::string name() const
        {
            return "sfc-" + m_curve.name();
        }

        /**
         * Destination rank of each cell by equal-weight cuts of the curve.
         * @note MPI: O(1) all_reduce for the setup + <= 64 all_reduce of P-1
         *       doubles for the cut search (collective), no data gather.
         */
        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& weight) const
        {
            using mesh_id_t           = typename Mesh::mesh_id_t;
            using cell_t              = typename Mesh::cell_t;
            constexpr std::size_t dim = Mesh::dim;

            boost::mpi::communicator world;
            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());

            // normalization level: the *config* max level, identical on every
            // rank (the local max level is not!)
            const std::size_t max_level = mesh.max_level();

            // -- 1a. global bounding box: curves need non-negative input, and the
            // per-dimension extent lets the curve preserve locality on non-square
            // domains (Hilbert lays a generalized curve over the exact box; a
            // square mapping fractures a thin domain's partitions into islands) --
            std::array<std::int64_t, dim> global_min;
            xt::xtensor_fixed<std::int64_t, xt::xshape<dim>> extent; // box size per dim, normalized to max_level

            if constexpr (requires { mesh.domain(); })
            {
                const auto coord_minmax = mesh.domain().minmax_indices();
                for (std::size_t d = 0; d < dim; ++d)
                {
                    global_min[d] = coord_minmax[d].first;
                    extent[d]     = coord_minmax[d].second - global_min[d] + 1; // >= 1
                }
            }
            else
            {
                const auto coord_minmax = mesh.minmax_indices();
                for (std::size_t d = 0; d < dim; ++d)
                {
                    global_min[d]              = boost::mpi::all_reduce(world, coord_minmax[d].first, boost::mpi::minimum<std::int64_t>());
                    const std::int64_t box_max = boost::mpi::all_reduce(world, coord_minmax[d].second, boost::mpi::maximum<std::int64_t>());
                    extent[d]                  = box_max - global_min[d] + 1; // >= 1
                }
            }

            // -- 1b/2. keys + local sort --------------------------------------------
            struct Item
            {
                sfc_key_t key;
                double w;
                cell_t cell;
            };

            std::vector<Item> items;
            items.reserve(mesh.nb_cells(mesh_id_t::cells));
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              const auto shift = max_level - cell.level;
                              xt::xtensor_fixed<std::uint32_t, xt::xshape<dim>> p;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  const std::int64_t c = (static_cast<std::int64_t>(cell.indices[d]) << shift) - global_min[d];
                                  assert(c >= 0 && c < (std::int64_t(1) << Curve::max_bits(dim))
                                         && "normalized coordinate exceeds the curve range: max_level too deep for 64-bit keys");
                                  p(d) = static_cast<std::uint32_t>(c);
                              }
                              items.push_back({m_curve.template key<dim>(p, extent), weight(cell), cell});
                          });

            std::sort(items.begin(),
                      items.end(),
                      [](const Item& a, const Item& b)
                      {
                          return a.key < b.key;
                      });

            // -- 3. total weight ----------------------------------------------------
            double w_local = 0.;
            for (const auto& item : items)
            {
                w_local += item.w;
            }
            const double w_total = boost::mpi::all_reduce(world, w_local, std::plus<double>());

            if (w_total <= 0.)
            {
                return flags; // nothing to balance on
            }

            // -- 4. equal-weight cuts with a *globally* correct cumulative ----------
            // A cell whose global exclusive-prefix weight along the curve is c
            // goes to rank floor(c / chunk), chunk = W_total / P. The prefix must
            // be the weight of *all* cells (every rank) with a smaller key.
            //
            // The previous version derived c from an MPI scan over the *rank
            // index* (offset = weight of the lower-index ranks). That equals the
            // curve prefix only if every rank already owns a curve-contiguous
            // segment. From the default decomposition (mesh.hpp partition_mesh
            // splits the intervals row-major, not along the curve) the local key
            // chunks of the ranks interleave, the scan offset is wrong, and the
            // partition fractures into disconnected pieces -- even for the
            // continuous Hilbert curve, because the *assignment* is globally
            // inconsistent, not the curve.
            //
            // Instead, locate the P-1 cut keys s_r = smallest key c such that the
            // global weight of the cells with key < c reaches r * chunk, by a
            // vectorised binary search over the 64-bit key space: each step
            // evaluates W_<(mid_r) locally (prefix sum + lower_bound on the
            // sorted keys) and sums it across ranks with one all_reduce. A cell
            // of key k then goes to rank #{ r : s_r <= k }. No cell/key gather:
            // O(64) all_reduce of (P-1) doubles, deterministic and exact.
            const int P        = world.size();
            const double chunk = w_total / static_cast<double>(P);

            std::vector<sfc_key_t> keys(items.size());
            std::vector<double> prefix(items.size() + 1, 0.);
            for (std::size_t i = 0; i < items.size(); ++i)
            {
                keys[i]       = items[i].key;
                prefix[i + 1] = prefix[i] + items[i].w;
            }
            const auto local_weight_below = [&](sfc_key_t c)
            {
                const auto it = std::lower_bound(keys.begin(), keys.end(), c);
                return prefix[static_cast<std::size_t>(it - keys.begin())];
            };

            const std::size_t n_cuts = (P > 0) ? static_cast<std::size_t>(P - 1) : 0;
            // search bracket [0, key_sup]: W_<(key_sup) == w_total > r * chunk
            const sfc_key_t local_max = keys.empty() ? sfc_key_t{0} : keys.back();
            const sfc_key_t key_sup   = boost::mpi::all_reduce(world, local_max, boost::mpi::maximum<sfc_key_t>()) + 1;

            std::vector<sfc_key_t> lo(n_cuts, 0);
            std::vector<sfc_key_t> hi(n_cuts, key_sup);
            std::vector<sfc_key_t> mid(n_cuts);
            std::vector<double> w_below(n_cuts);
            std::vector<double> w_below_global(n_cuts);
            for (int iter = 0; iter < 64; ++iter)
            {
                bool active = false;
                for (std::size_t r = 0; r < n_cuts; ++r)
                {
                    active |= (lo[r] < hi[r]);
                    mid[r]     = lo[r] + (hi[r] - lo[r]) / 2;
                    w_below[r] = local_weight_below(mid[r]);
                }
                if (!active)
                {
                    break;
                }
                boost::mpi::all_reduce(world, w_below.data(), static_cast<int>(n_cuts), w_below_global.data(), std::plus<double>());
                for (std::size_t r = 0; r < n_cuts; ++r)
                {
                    if (lo[r] >= hi[r])
                    {
                        continue;
                    }
                    if (w_below_global[r] >= static_cast<double>(r + 1) * chunk)
                    {
                        hi[r] = mid[r];
                    }
                    else
                    {
                        lo[r] = mid[r] + 1;
                    }
                }
            }

            // lo holds the cut keys s_1 <= ... <= s_{P-1}: rank(k) = #{ r : s_r <= k }
            for (const auto& item : items)
            {
                const auto rank  = std::upper_bound(lo.begin(), lo.end(), item.key) - lo.begin();
                flags[item.cell] = static_cast<int>(rank);
            }
            return flags;
        }

      private:

        Curve m_curve;
    };
}
#endif
