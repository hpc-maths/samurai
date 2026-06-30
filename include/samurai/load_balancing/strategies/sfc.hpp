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
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
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
    /**
     * Global bounding box of the mesh, normalized to its max level.
     *
     * Single source of truth shared by the SFC strategy and its tests so the
     * key normalization cannot drift apart. `minmax_indices()` is half-open
     * (`.first` inclusive, `.second` exclusive), hence the per-dimension box
     * size is `second - first`. The per-dimension extent lets Hilbert lay a
     * generalized curve over the exact box, preserving locality on non-square
     * domains (a square mapping fractures a thin domain into islands).
     *
     * @return {global_min, extent}: the shift making coordinates non-negative
     *         (curves require >= 0) and the box size per dimension.
     */
    template <class Mesh>
    auto sfc_normalized_box(const Mesh& mesh)
    {
        constexpr std::size_t dim = Mesh::dim;
        std::array<std::int64_t, dim> global_min;
        xt::xtensor_fixed<std::int64_t, xt::xshape<dim>> extent; // box size per dim, normalized to max_level

        if constexpr (requires { mesh.domain(); })
        {
            const auto coord_minmax = mesh.domain().minmax_indices();
            for (std::size_t d = 0; d < dim; ++d)
            {
                global_min[d] = coord_minmax[d].first;
                extent[d]     = coord_minmax[d].second - global_min[d]; // >= 1
            }
        }
        else
        {
            boost::mpi::communicator world;
            const auto coord_minmax = mesh.minmax_indices();
            for (std::size_t d = 0; d < dim; ++d)
            {
                global_min[d]              = boost::mpi::all_reduce(world, coord_minmax[d].first, boost::mpi::minimum<std::int64_t>());
                const std::int64_t box_max = boost::mpi::all_reduce(world, coord_minmax[d].second, boost::mpi::maximum<std::int64_t>());
                extent[d]                  = box_max - global_min[d]; // >= 1
            }
        }
        return std::make_pair(global_min, extent);
    }

    /**
     * SFC key of a cell, normalized to `max_level` and shifted into the box
     * returned by sfc_normalized_box. Single source of truth for the key so the
     * strategy and its validators stay in lock-step.
     */
    template <class Curve, class Cell, class GlobalMin, class Extent>
    sfc_key_t sfc_cell_key(const Curve& curve, const Cell& cell, std::size_t max_level, const GlobalMin& global_min, const Extent& extent)
    {
        constexpr std::size_t dim = Cell::dim;
        const auto shift          = max_level - cell.level;
        xt::xtensor_fixed<std::uint32_t, xt::xshape<dim>> p;
        for (std::size_t d = 0; d < dim; ++d)
        {
            const std::int64_t c = (static_cast<std::int64_t>(cell.indices[d]) << shift) - global_min[d];
            assert(c >= 0 && c < (std::int64_t(1) << Curve::max_bits(dim))
                   && "normalized coordinate exceeds the curve range: max_level too deep for 64-bit keys");
            p(d) = static_cast<std::uint32_t>(c);
        }
        return curve.template key<dim>(p, extent);
    }

    /**
     * P-1 equal-weight cut keys via a gather-free vectorised binary search over
     * the key space: each step evaluates W_<(mid) locally (prefix sum +
     * lower_bound on the sorted keys) and sums it across ranks with one
     * all_reduce. Used for cell atoms (too many to gather). Deterministic and
     * exact from any initial decomposition.
     *
     * @param sorted_keys local atom keys, ascending; @param weights aligned with
     * them. @param chunk = W_total / P. Returns the P-1 cut keys, identical on
     * every rank: an atom of key k goes to rank #{ r : lo[r] <= k }.
     */
    inline std::vector<sfc_key_t> sfc_equal_weight_cuts_search(const std::vector<sfc_key_t>& sorted_keys,
                                                               const std::vector<double>& weights,
                                                               int P,
                                                               double chunk,
                                                               const boost::mpi::communicator& world)
    {
        const std::size_t n_cuts = (P > 0) ? static_cast<std::size_t>(P - 1) : 0;

        std::vector<double> prefix(sorted_keys.size() + 1, 0.);
        for (std::size_t i = 0; i < sorted_keys.size(); ++i)
        {
            prefix[i + 1] = prefix[i] + weights[i];
        }
        const auto local_weight_below = [&](sfc_key_t c)
        {
            const auto it = std::lower_bound(sorted_keys.begin(), sorted_keys.end(), c);
            return prefix[static_cast<std::size_t>(it - sorted_keys.begin())];
        };

        // search bracket [0, key_sup]: W_<(key_sup) == w_total > r * chunk
        const sfc_key_t local_max = sorted_keys.empty() ? sfc_key_t{0} : sorted_keys.back();
        const auto global_max     = boost::mpi::all_reduce(world, local_max, boost::mpi::maximum<sfc_key_t>());
        const sfc_key_t key_sup   = (global_max == std::numeric_limits<sfc_key_t>::max()) ? global_max : global_max + 1;
        ;

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
        return lo;
    }

    /**
     * Same P-1 cut keys as sfc_equal_weight_cuts_search, but a single all_gatherv
     * of every (key, weight) replaces the iterated collective: all ranks build
     * the global key-sorted sequence and find the cuts in one local sweep. For
     * interval atoms only, where the global atom count is small enough to gather.
     * Atoms sharing a key are accumulated as one block so the result is
     * bit-identical to the search. @param keys / @param weights need not be
     * locally sorted; @param counts holds each rank's atom count.
     */
    inline std::vector<sfc_key_t> sfc_equal_weight_cuts_gather(const std::vector<sfc_key_t>& keys,
                                                               const std::vector<double>& weights,
                                                               const std::vector<int>& counts,
                                                               int P,
                                                               double chunk,
                                                               const boost::mpi::communicator& world)
    {
        const std::size_t n_cuts = (P > 0) ? static_cast<std::size_t>(P - 1) : 0;

        std::vector<sfc_key_t> gkeys;
        std::vector<double> gw;
        boost::mpi::all_gatherv(world, keys, gkeys, counts);
        boost::mpi::all_gatherv(world, weights, gw, counts);

        std::vector<std::size_t> order(gkeys.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(),
                  order.end(),
                  [&](std::size_t a, std::size_t b)
                  {
                      return gkeys[a] < gkeys[b];
                  });

        // sweep the key-sorted atoms by block (equal keys grouped); the cut s_r is
        // k+1 at the first block whose inclusive prefix reaches (r+1)*chunk --
        // identical to the search's smallest c with W_<(c) >= T.
        const sfc_key_t key_sup = gkeys.empty() ? sfc_key_t{0} : (gkeys[order.back()] + 1);
        std::vector<sfc_key_t> lo(n_cuts, key_sup);
        double pre    = 0.;
        std::size_t r = 0;
        std::size_t t = 0;
        while (t < order.size() && r < n_cuts)
        {
            const sfc_key_t k = gkeys[order[t]];
            double block      = 0.;
            while (t < order.size() && gkeys[order[t]] == k)
            {
                block += gw[order[t]];
                ++t;
            }
            pre += block;
            while (r < n_cuts && pre >= static_cast<double>(r + 1) * chunk)
            {
                lo[r] = k + 1;
                ++r;
            }
        }
        return lo;
    }

    template <class Curve>
    class SFC
    {
      public:

        SFC() = default;

        explicit SFC(Curve curve, bool by_interval = false)
            : m_curve(std::move(curve))
            , m_by_interval(by_interval)
        {
        }

        /**
         * Choose the partition atom.
         *  - false (default): one atom per cell.
         *  - true: one atom per x-interval, keyed on its first cell and weighted
         *    by the whole interval; the interval is never split across ranks.
         * Interval atoms are ~10-20x fewer (advection 2D ~11x, 3D ~19x), so the
         * cut search runs on far fewer keys; the trade-off is a coarser
         * granularity (an interval is indivisible -- harmless until the heaviest
         * interval approaches W_total / P, i.e. very large process counts).
         */
        SFC& with_interval_atoms(bool v = true)
        {
            m_by_interval = v;
            return *this;
        }

        bool uses_interval_atoms() const
        {
            return m_by_interval;
        }

        std::string name() const
        {
            return "sfc-" + m_curve.name() + (m_by_interval ? "-interval" : "-cell");
        }

        /**
         * Destination rank of each cell by equal-weight cuts of the curve.
         * @note MPI: O(1) all_reduce for the setup + <= 64 all_reduce of P-1
         *       doubles for the cut search (collective), no data gather.
         */
        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& weight) const
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            using cell_t    = typename Mesh::cell_t;

            boost::mpi::communicator world;
            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());

            // normalization level: the *config* max level, identical on every
            // rank (the local max level is not!)
            const std::size_t max_level = mesh.max_level();

            // -- 1a. global bounding box (shift + per-dimension extent), shared
            // with the partition validators via sfc_normalized_box. --
            const auto [global_min, extent] = sfc_normalized_box(mesh);

            // -- 1b/2. keys + local sort --------------------------------------------
            struct Item
            {
                sfc_key_t key;
                double w;
                cell_t cell;
            };

            std::vector<Item> items;

            constexpr std::size_t dim = Mesh::dim;
            auto& ca                  = mesh[mesh_id_t::cells];
            const auto origin         = ca.origin_point();
            const double sf           = ca.scaling_factor();

            // Visit each x-interval, exposing its first cell as representative and
            // a factory `make_cell(x)` for the cell at column x. Same O(1)
            // construction as for_each_cell (interval storage offset iv.index),
            // but grouped by interval so atoms can be whole intervals.
            auto for_each_x_interval = [&](auto&& on_interval)
            {
                for_each_interval(ca,
                                  [&](std::size_t level, const auto& iv, const auto& index_yz)
                                  {
                                      typename cell_t::indices_t indices{};
                                      for (std::size_t k = 0; k + 1 < dim; ++k)
                                      {
                                          indices[k + 1] = index_yz[k];
                                      }
                                      auto make_cell = [=](auto x)
                                      {
                                          typename cell_t::indices_t id = indices;
                                          id[0]                         = x;
                                          return cell_t{origin, sf, level, id, iv.index + x};
                                      };
                                      on_interval(iv, make_cell);
                                  });
            };

            if (m_by_interval)
            {
                // atoms = x-intervals: key from the first cell, weight summed over
                // the interval.
                for_each_x_interval(
                    [&](const auto& iv, auto make_cell)
                    {
                        const auto rep = make_cell(iv.start);
                        double w       = 0.;
                        for (auto x = iv.start; x < iv.end; ++x)
                        {
                            w += weight(make_cell(x));
                        }
                        items.push_back({sfc_cell_key(m_curve, rep, max_level, global_min, extent), w, rep});
                    });
            }
            else
            {
                // atoms = cells (default).
                items.reserve(mesh.nb_cells(mesh_id_t::cells));
                for_each_cell(ca,
                              [&](const auto& cell)
                              {
                                  items.push_back({sfc_cell_key(m_curve, cell, max_level, global_min, extent), weight(cell), cell});
                              });
            }

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

            // -- 4. equal-weight cuts of the curve ---------------------------------
            // The P-1 cut keys s_r = smallest key c such that the *global* weight
            // of the atoms with key < c reaches r * chunk (chunk = W_total / P).
            // An atom of key k then goes to rank #{ r : s_r <= k }, i.e.
            // floor(c / chunk) with c its global exclusive-prefix weight. The cut
            // keys depend only on the global cumulative weight, never on which
            // rank currently owns an atom: correct from *any* decomposition (an
            // MPI scan over the rank index would instead need each rank to already
            // own a curve-contiguous segment, which the row-major default
            // decomposition does not provide).
            //
            // Two equivalent ways to obtain the same cut keys:
            //  - cell atoms: too many to gather, so a gather-free vectorised
            //    binary search over the key space (sfc_equal_weight_cuts_search,
            //    <=64 all_reduce of P-1 doubles);
            //  - interval atoms: ~10-20x fewer, so a single all_gatherv of every
            //    (key, weight) + a local sweep (sfc_equal_weight_cuts_gather),
            //    dropping the iterated collective. A guard falls back to the
            //    search if the global atom count is unexpectedly large.
            const int P        = world.size();
            const double chunk = w_total / static_cast<double>(P);

            std::vector<sfc_key_t> keys(items.size());
            std::vector<double> w(items.size());
            for (std::size_t i = 0; i < items.size(); ++i)
            {
                keys[i] = items[i].key; // items are key-sorted (step 2)
                w[i]    = items[i].w;
            }

            std::vector<sfc_key_t> lo;
            bool cuts_done = false;
            if (m_by_interval)
            {
                std::vector<int> counts;
                boost::mpi::all_gather(world, static_cast<int>(items.size()), counts);
                const long long n_global = std::accumulate(counts.begin(), counts.end(), 0LL);
                if (n_global <= static_cast<long long>(gather_atom_cap))
                {
                    lo        = sfc_equal_weight_cuts_gather(keys, w, counts, P, chunk, world);
                    cuts_done = true;
                }
            }
            if (!cuts_done)
            {
                lo = sfc_equal_weight_cuts_search(keys, w, P, chunk, world);
            }

            // lo holds the cut keys s_1 <= ... <= s_{P-1}: rank(k) = #{ r : s_r <= k }
            const auto rank_of = [&](sfc_key_t key)
            {
                return static_cast<int>(std::upper_bound(lo.begin(), lo.end(), key) - lo.begin());
            };

            if (m_by_interval)
            {
                // every cell of an interval inherits its representative's rank, so
                // an interval is never split across ranks.
                for_each_x_interval(
                    [&](const auto& iv, auto make_cell)
                    {
                        const int rank = rank_of(sfc_cell_key(m_curve, make_cell(iv.start), max_level, global_min, extent));
                        for (auto x = iv.start; x < iv.end; ++x)
                        {
                            flags[make_cell(x)] = rank;
                        }
                    });
            }
            else
            {
                for (const auto& item : items)
                {
                    flags[item.cell] = rank_of(item.key);
                }
            }
            return flags;
        }

      private:

        // Above this many *global* atoms the gather path is abandoned for the
        // gather-free search (interval atoms stay well below this in practice).
        static constexpr std::size_t gather_atom_cap = 4'000'000;

        Curve m_curve;
        bool m_by_interval = false;
    };
}
#endif
