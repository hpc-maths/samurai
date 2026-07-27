// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>

#include <xtensor/containers/xfixed.hpp>

#include "algorithm.hpp"
#include "level_cell_array.hpp"
#include "numeric/prediction_coefficients.hpp"

/**
 * @file prediction_shifts.hpp
 *
 * Where a prediction stencil sits, read off the domain.
 *
 * numeric/prediction_coefficients.hpp holds the numeric half of the question - given a
 * shift, what are the coefficients - and knows nothing about a mesh. This file holds the
 * geometric half: given a cell, what shift must the stencil take so that it reads only
 * cells the domain has. The two are kept apart on purpose, because the numeric half is a
 * global memo that must stay mesh-free.
 *
 * **The domain, never the subdomain.** The shift is classified against the *global,
 * replicated* domain at the level in question, and never against the cells one MPI rank
 * happens to hold. That is what makes the operator partition independent with no
 * communication: every rank asks the same question of the same object and gets the same
 * answer. Whether the cells the chosen stencil names are present *locally* is a separate,
 * halo question, answered by the ghost width, and it is not this file's business.
 * Conflating the two - "clamp to what I can see" - is the one way to make results depend
 * on the rank count.
 *
 * **One query per interval, not per cell.** @ref for_each_prediction_shift_run walks an
 * interval once and hands back the maximal runs over which the shift is constant, so the
 * consumer keeps its hoists: it selects a coefficient table per run and leaves its inner
 * loop untouched. On a box domain an interval away from the boundary yields exactly one
 * run with all shifts zero, which is what keeps interior values bit-identical.
 *
 * The decomposition is exact on a holed domain, and that is the reason it is a run
 * decomposition rather than one class per interval: an interval passing over the edge of a
 * hole has cells whose stencil must be shifted and cells whose stencil must not, and
 * classifying the whole interval by its worst cell would move interior values.
 *
 * **A periodic direction has no boundary.** Stepping off the end of one reaches the cells
 * the periodic ghost exchange fills from, so the stencil stays centred there and nothing is
 * clamped; the caller says which directions those are by passing their wrap. Holes are
 * unaffected by this, in a periodic direction as in any other, because only stepping off
 * the end of the domain wraps.
 */

namespace samurai
{
    /**
     * The prediction stencil shifts holding over a run of cells: one shift per direction.
     */
    template <std::size_t dim>
    struct PredictionShifts
    {
        std::array<int, dim> shift{}; ///< @c shift[d] is what @ref prediction_coefficients takes along @c d
        bool fits = true;             ///< false where no shift makes the stencil fit (see @ref prediction_shift)
    };

    namespace detail
    {
        /**
         * A cursor over the x-intervals of one row of a level cell array - the cells at a
         * fixed transverse index. Rows are queried at increasing x, so the cursor only
         * moves forward and scanning a whole interval costs one pass over the intervals it
         * meets rather than a search per cell.
         */
        template <class TInterval>
        class RowScan
        {
          public:

            using value_t = typename TInterval::value_t;

            RowScan() = default;

            RowScan(const TInterval* first, std::size_t size)
                : m_first(first)
                , m_size(size)
            {
            }

            /**
             * Does this row hold the cell at @a x? @a next_change is set to the smallest
             * coordinate above @a x at which the answer changes, so a caller scanning a
             * range knows how far the answer it just got remains valid.
             */
            bool covers(value_t x, value_t& next_change)
            {
                while (m_cursor < m_size && m_first[m_cursor].end <= x)
                {
                    ++m_cursor;
                }

                if (m_cursor == m_size)
                {
                    next_change = std::numeric_limits<value_t>::max();
                    return false;
                }
                if (m_first[m_cursor].start <= x)
                {
                    next_change = m_first[m_cursor].end;
                    return true;
                }
                next_change = m_first[m_cursor].start;
                return false;
            }

            /// True when the domain has no cell at this transverse index at all.
            bool empty() const
            {
                return m_size == 0;
            }

            /// The interval holding the last @c x that @ref covers accepted.
            const TInterval& current() const
            {
                return m_first[m_cursor];
            }

            /**
             * Does this row hold the cell at @a x, asked out of order? The cursor above is
             * for scanning; this is for the handful of cells a periodic wrap reaches at the
             * far end of the row.
             */
            bool contains(value_t x) const
            {
                const auto* it = std::upper_bound(m_first,
                                                  m_first + m_size,
                                                  x,
                                                  [](value_t v, const TInterval& itv)
                                                  {
                                                      return v < itv.start;
                                                  });
                return it != m_first && (it - 1)->end > x;
            }

          private:

            const TInterval* m_first = nullptr;
            std::size_t m_size       = 0;
            std::size_t m_cursor     = 0;
        };

        /**
         * The row of @a lca at transverse index @a index, as a cursor. Empty when the
         * domain has no cell at that index at all.
         */
        template <std::size_t dim, class TInterval>
        RowScan<TInterval>
        row_scan(const LevelCellArray<dim, TInterval>& lca, const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim - 1>>& index)
        {
            std::size_t start = 0;
            std::size_t end   = lca[dim - 1].size();

            // The transverse dimensions are nested outermost first, so the range of
            // x-intervals is narrowed one dimension at a time, as find() does.
            for (std::size_t d = dim - 1; d >= 1; --d)
            {
                const auto pos = find_on_dim(lca, d, start, end, index[d - 1]);
                if (pos == std::numeric_limits<std::size_t>::max())
                {
                    return {};
                }
                const auto offset = static_cast<std::size_t>(lca[d][pos].index + index[d - 1]);
                start             = lca.offsets(d)[offset];
                end               = lca.offsets(d)[offset + 1];
            }

            if (start == end)
            {
                return {};
            }
            return {lca[0].data() + start, end - start};
        }
    }

    /**
     * Split @a i into the maximal runs over which the prediction stencil shift is
     * constant, and call @a f on each as @c f(run, shifts).
     *
     * @tparam radius prediction stencil radius
     * @param domain  the domain at the level @a i lives at - global and replicated, see
     *                the file comment
     * @param period  the periodic wrap of each direction, at that level, and @c 0 where the
     *                direction is not periodic. It is the same quantity the periodic ghost
     *                exchange shifts by, @c (max_indices[d] - min_indices[d]) >> delta_l,
     *                and it must agree with it: a periodic direction has no boundary, so
     *                clamping a stencil against one would move values for nothing.
     * @param i       an interval of cells, at that same level
     * @param index   its transverse index
     *
     * Each run carries the storage index of @a i, so a consumer can index a field with it
     * exactly as it indexes @a i.
     *
     * Cells of @a i that the domain does not hold are reported as their own runs with
     * @c fits false, whether or not the direction is periodic: prediction has nothing to
     * reach into there. The query reports it rather than deciding what to do about it.
     */
    template <std::size_t radius, std::size_t dim, class TInterval, class Func>
    void for_each_prediction_shift_run(const LevelCellArray<dim, TInterval>& domain,
                                       const std::array<typename TInterval::value_t, dim>& period,
                                       const TInterval& i,
                                       const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim - 1>>& index,
                                       Func&& f)
    {
        using value_t = typename TInterval::value_t;

        // Availability is only ever needed up to 2r: a stencil short by r on one side needs
        // 2r available opposite it, and nothing beyond that changes the answer.
        constexpr auto reach = static_cast<value_t>(2 * radius);

        auto own = detail::row_scan(domain, index);

        // The rows the transverse directions read, indexed [direction][side][distance - 1].
        std::array<std::array<std::array<detail::RowScan<TInterval>, 2 * radius>, 2>, dim - 1> rows;
        for (std::size_t d = 0; d + 1 < dim; ++d)
        {
            for (std::size_t side = 0; side < 2; ++side)
            {
                for (std::size_t k = 1; k <= 2 * radius; ++k)
                {
                    auto neighbour = index;
                    neighbour[d] += (side == 0 ? -1 : 1) * static_cast<value_t>(k);
                    auto row = detail::row_scan(domain, neighbour);

                    // Off the end of a periodic direction, the cells the stencil reads are
                    // the ones the periodic exchange fills it from, one wrap away. A row
                    // that exists is never wrapped, so a hole inside the domain still
                    // clamps: only stepping off the end wraps.
                    if (row.empty() && period[d + 1] != 0)
                    {
                        neighbour[d] -= (side == 0 ? -1 : 1) * period[d + 1];
                        row = detail::row_scan(domain, neighbour);
                    }
                    rows[d][side][k - 1] = row;
                }
            }
        }

        value_t x = i.start;
        while (x < i.end)
        {
            value_t next   = i.end;
            value_t change = 0;

            const bool inside = own.covers(x, change);
            next              = std::min(next, change);

            if (!inside)
            {
                f(TInterval{x, next, i.index}, PredictionShifts<dim>{{}, false});
                x = next;
                continue;
            }

            std::array<int, dim> avail_low{};
            std::array<int, dim> avail_high{};

            const auto& own_interval = own.current();
            avail_low[0]             = static_cast<int>(std::min(x - own_interval.start, reach));
            avail_high[0]            = static_cast<int>(std::min(own_interval.end - 1 - x, reach));

            // In x the class changes cell by cell within 2r of each end of the run of cells
            // the domain holds, and is constant in between - the bulk the consumers keep
            // their current kernel on. This reads the geometry, before any periodic wrap
            // below tops the counts back up: a wrap makes the two ends read the same as the
            // bulk, it does not make them one run with it.
            if (avail_low[0] < reach || avail_high[0] < reach)
            {
                next = std::min(next, x + 1);
            }
            else
            {
                next = std::min(next, own_interval.end - reach);
            }

            // Same wrap as the transverse rows, one cell at a time because in x the count
            // varies from cell to cell. A cell missing because of a hole is not restored by
            // this: its image one wrap away is outside the domain, so the test fails.
            if (period[0] != 0)
            {
                for (auto k = avail_low[0]; k < reach && own.contains(x - k - 1 + period[0]); ++k)
                {
                    avail_low[0] = k + 1;
                }
                for (auto k = avail_high[0]; k < reach && own.contains(x + k + 1 - period[0]); ++k)
                {
                    avail_high[0] = k + 1;
                }
            }

            for (std::size_t d = 0; d + 1 < dim; ++d)
            {
                for (std::size_t side = 0; side < 2; ++side)
                {
                    int available = 0;
                    for (std::size_t k = 0; k < 2 * radius; ++k)
                    {
                        value_t row_change = 0;
                        const bool covered = rows[d][side][k].covers(x, row_change);
                        next               = std::min(next, row_change);
                        if (!covered)
                        {
                            // Rows further out cannot restore what this one denies, and
                            // they cannot change the count while this one is missing.
                            break;
                        }
                        ++available;
                    }
                    (side == 0 ? avail_low : avail_high)[d + 1] = available;
                }
            }

            PredictionShifts<dim> shifts;
            for (std::size_t d = 0; d < dim; ++d)
            {
                const auto shift = prediction_shift(radius, avail_low[d], avail_high[d]);
                shifts.shift[d]  = shift.shift;
                shifts.fits      = shifts.fits && shift.fits;
            }

            // A breakpoint that failed to move would spin here rather than fail, so it is
            // asserted instead of being clamped away.
            assert(next > x && "for_each_prediction_shift_run: the run decomposition did not advance");

            f(TInterval{x, next, i.index}, shifts);
            x = next;
        }
    }

    /**
     * The prediction stencil shifts at one cell. The per-interval form above is what the
     * hot kernels use; this is for consumers that already visit cells one at a time, and
     * for saying in one line what a test means.
     */
    template <std::size_t radius, std::size_t dim, class TInterval>
    PredictionShifts<dim> prediction_shifts_at(const LevelCellArray<dim, TInterval>& domain,
                                               const std::array<typename TInterval::value_t, dim>& period,
                                               const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim>>& coord)
    {
        using value_t = typename TInterval::value_t;

        xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> index;
        for (std::size_t d = 0; d + 1 < dim; ++d)
        {
            index[d] = coord[d + 1];
        }

        PredictionShifts<dim> shifts;
        for_each_prediction_shift_run<radius>(domain,
                                              period,
                                              TInterval{coord[0], coord[0] + 1},
                                              index,
                                              [&](const auto&, const auto& run_shifts)
                                              {
                                                  shifts = run_shifts;
                                              });
        return shifts;
    }
}
