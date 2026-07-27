// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <limits>

#include <xtensor/containers/xfixed.hpp>

#include "algorithm.hpp"
#include "level_cell_array.hpp"

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
 * **The stencil is a box, so the question is asked of the box.** The consumers apply the 1D
 * family as a tensor product, so the cells a stencil reads at a cell @c c with shift @c s
 * are the whole box `prod_d [c_d + s_d - r, c_d + s_d + r]` - the mixed terms included. A
 * shift is admissible exactly when the domain holds that entire box, and the answer is the
 * *most centred* admissible shift:
 *
 *   1. the least shifted overall, i.e. the smallest `sum_d |s_d|` - a shift is what makes a
 *      stencil one-sided, so the fewer cells it moves in total the better;
 *   2. among those, the one shifting @c x least, then @c y, and so on: a shift along @c x
 *      moves the innermost loop's reads, while a transverse shift only picks a different
 *      row, so a deficit is better absorbed transversally;
 *   3. among those, the negative shift. Nothing distinguishes the two, so the tie is broken
 *      by fiat rather than left to the traversal order - the answer must not depend on how
 *      the domain happens to be stored.
 *
 * Asking the box, rather than each direction separately, is what a **re-entrant corner**
 * needs. Per-direction availability cannot see a cell that is missing only diagonally: at
 * the cell diagonally off the corner of a hole, every direction reads cells the domain has,
 * yet the corner of the box is inside the hole. The two rules agree everywhere else - on a
 * box domain they agree at every cell, corners included - so this is a statement about
 * re-entrant corners only, whether they belong to a hole or to an L-shaped domain.
 *
 * That also makes @c fits a joint condition: it is false when *no* shift makes the box fit,
 * which is strictly stronger than each direction being wide enough on its own. It is the
 * constructibility condition the mesh has to guarantee, and a caller is expected to report
 * it loudly rather than silently degrade the order.
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
 * interval once and hands back the maximal runs over which the shift is constant - maximal
 * in the strict sense that no two runs it emits are adjacent with the same shift - so the
 * consumer keeps its hoists: it selects a coefficient table per run and leaves its inner
 * loop untouched. On a box domain an interval away from the boundary yields exactly one run
 * with all shifts zero, which is what keeps interior values bit-identical.
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
 *
 * **Cost.** One query rescans the `(4r+1)^(dim-1)` transverse rows the stencil can reach,
 * so a consumer sweeping row by row repeats the transverse searches of its neighbours. That
 * is the first thing to look at should this appear in a profile; the fix would be a cache
 * keyed on the transverse index, not a change of rule.
 */

namespace samurai
{
    /**
     * Where the prediction stencil sits over a run of cells: one shift per direction.
     */
    template <std::size_t dim>
    struct PredictionStencilShift
    {
        std::array<int, dim> shift{}; ///< @c shift[d] is what @c prediction_coefficients takes along @c d
        bool fits = true;             ///< false where no shift makes the stencil box fit (see the file comment)

        bool operator==(const PredictionStencilShift& o) const
        {
            return shift == o.shift && fits == o.fits;
        }
    };

    namespace detail
    {
        constexpr std::size_t ipow(std::size_t base, std::size_t exponent)
        {
            std::size_t out = 1;
            for (std::size_t k = 0; k < exponent; ++k)
            {
                out *= base;
            }
            return out;
        }

        /**
         * The @a n-th vector of the box `[-half, half]^len`, direction 0 varying fastest.
         * The inverse is the mixed-radix digit sum, which is how the tables below index
         * rows.
         */
        template <std::size_t len>
        constexpr std::array<int, len> nth_offset(std::size_t n, int half)
        {
            const auto extent = static_cast<std::size_t>(2 * half + 1);

            std::array<int, len> out{};
            for (std::size_t d = 0; d < len; ++d)
            {
                out[d] = static_cast<int>(n % extent) - half;
                n /= extent;
            }
            return out;
        }

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

        /**
         * The row of @a domain at the transverse index @a index displaced by @a offset,
         * wrapped where the displacement has stepped off the end of a periodic direction.
         *
         * A row that exists is never wrapped, so a hole inside the domain still clamps:
         * only stepping off the *end* of the domain wraps. Two transverse directions can
         * step off at once - the corner of a doubly periodic 3D domain - and wrapping one
         * of them alone leaves the row still empty, so the combinations are tried fewest
         * wraps first and the first non-empty one wins.
         */
        template <std::size_t dim, class TInterval>
        RowScan<TInterval> wrapped_row_scan(const LevelCellArray<dim, TInterval>& domain,
                                            const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim - 1>>& index,
                                            const std::array<int, dim - 1>& offset,
                                            const std::array<typename TInterval::value_t, dim>& period)
        {
            using value_t               = typename TInterval::value_t;
            constexpr std::size_t cross = dim - 1;

            auto base = index;
            for (std::size_t d = 0; d < cross; ++d)
            {
                base[d] += static_cast<value_t>(offset[d]);
            }

            auto row = row_scan(domain, base);
            if (!row.empty())
            {
                return row;
            }

            for (std::size_t wraps = 1; wraps <= cross; ++wraps)
            {
                for (std::size_t mask = 0; mask < (std::size_t{1} << cross); ++mask)
                {
                    std::size_t bits = 0;
                    for (std::size_t d = 0; d < cross; ++d)
                    {
                        bits += (mask >> d) & std::size_t{1};
                    }
                    if (bits != wraps)
                    {
                        continue;
                    }

                    auto wrapped = base;
                    bool usable  = true;
                    for (std::size_t d = 0; d < cross && usable; ++d)
                    {
                        if (((mask >> d) & std::size_t{1}) == 0)
                        {
                            continue;
                        }
                        // Only a direction the displacement stepped in can have stepped off
                        // the end, and only a periodic one wraps.
                        usable = offset[d] != 0 && period[d + 1] != 0;
                        if (usable)
                        {
                            wrapped[d] -= static_cast<value_t>(offset[d] > 0 ? 1 : -1) * period[d + 1];
                        }
                    }
                    if (!usable)
                    {
                        continue;
                    }

                    auto candidate = row_scan(domain, wrapped);
                    if (!candidate.empty())
                    {
                        return candidate;
                    }
                }
            }
            return row;
        }

        /// Is @a a the more centred of the two shifts? The order is the file comment's.
        template <std::size_t dim>
        bool more_centred(const std::array<int, dim>& a, const std::array<int, dim>& b)
        {
            int total_a = 0;
            int total_b = 0;
            for (std::size_t d = 0; d < dim; ++d)
            {
                total_a += std::abs(a[d]);
                total_b += std::abs(b[d]);
            }
            if (total_a != total_b)
            {
                return total_a < total_b;
            }
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (std::abs(a[d]) != std::abs(b[d]))
                {
                    return std::abs(a[d]) < std::abs(b[d]);
                }
            }
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (a[d] != b[d])
                {
                    return a[d] < b[d];
                }
            }
            return false;
        }

        /// How many transverse offsets a radius-@c radius stencil can reach: `[-2r, 2r]` per
        /// transverse direction, a shift of at most r reading at most r further out.
        template <std::size_t radius>
        constexpr std::size_t row_extent = 4 * radius + 1;

        /// The `[-2r, 2r]^(dim-1)` transverse offsets, flattened.
        template <std::size_t radius, std::size_t dim>
        constexpr std::size_t row_count = ipow(row_extent<radius>, dim - 1);

        /// The transverse rows one stencil box covers: `[-r, r]^(dim-1)` around its shift.
        template <std::size_t radius, std::size_t dim>
        constexpr std::size_t band_size = ipow(2 * radius + 1, dim - 1);

        /// The shifts a stencil can take: `[-r, r]^dim`.
        template <std::size_t radius, std::size_t dim>
        constexpr std::size_t shift_count = ipow(2 * radius + 1, dim);

        /**
         * The candidate shifts in the order they are preferred, with the rows each one
         * reads. Both depend on nothing but @c radius and @c dim, so they are tabulated
         * once: taking the first admissible entry of @c shift is the rule, and @c rows[c]
         * holds the indices - into the caller's `[-2r, 2r]^(dim-1)` row table - of the
         * `(2r+1)^(dim-1)` rows the c-th stencil box covers.
         */
        template <std::size_t radius, std::size_t dim>
        struct ShiftSearch
        {
            std::array<std::array<int, dim>, shift_count<radius, dim>> shift{};
            std::array<std::array<std::size_t, band_size<radius, dim>>, shift_count<radius, dim>> rows{};
        };

        template <std::size_t radius, std::size_t dim>
        const ShiftSearch<radius, dim>& shift_search()
        {
            constexpr int r             = static_cast<int>(radius);
            constexpr std::size_t cross = dim - 1;

            static const ShiftSearch<radius, dim> table = []
            {
                ShiftSearch<radius, dim> out;
                for (std::size_t c = 0; c < shift_count<radius, dim>; ++c)
                {
                    out.shift[c] = nth_offset<dim>(c, r);
                }
                std::sort(out.shift.begin(), out.shift.end(), more_centred<dim>);

                for (std::size_t c = 0; c < shift_count<radius, dim>; ++c)
                {
                    for (std::size_t k = 0; k < band_size<radius, dim>; ++k)
                    {
                        const auto within = nth_offset<cross>(k, r);

                        std::size_t flat   = 0;
                        std::size_t stride = 1;
                        for (std::size_t d = 0; d < cross; ++d)
                        {
                            const auto transverse = out.shift[c][d + 1] + within[d] + 2 * r;
                            flat += static_cast<std::size_t>(transverse) * stride;
                            stride *= row_extent<radius>;
                        }
                        out.rows[c][k] = flat;
                    }
                }
                return out;
            }();

            return table;
        }
    }

    /**
     * Split @a i into the maximal runs over which the prediction stencil shift is
     * constant, and call @a f on each as @c f(run, shift).
     *
     * @tparam radius prediction stencil radius
     * @param domain  the domain at the level @a i lives at - global and replicated, see
     *                the file comment
     * @param period  the periodic wrap of each direction, at that level, and @c 0 where the
     *                direction is not periodic. It is the same quantity the periodic ghost
     *                exchange shifts by, @c (max_indices[d] - min_indices[d]) >> delta_l,
     *                and it must agree with it: a periodic direction has no boundary, so
     *                clamping a stencil against one would move values for nothing. A wrap
     *                is applied at most once per direction, which is exact as long as the
     *                domain is at least @c 2*radius cells wide there - narrower than that,
     *                a stencil would read the same cell twice and the question is moot.
     * @param i       an interval of cells, at that same level
     * @param index   its transverse index
     *
     * Each run carries the storage index of @a i, so a consumer can index a field with it
     * exactly as it indexes @a i.
     *
     * Cells of @a i that the domain does not hold are reported with @c fits false, whether
     * or not the direction is periodic: prediction has nothing to reach into there. So are
     * cells the domain holds but around which no shift makes the stencil box fit. The query
     * reports it rather than deciding what to do about it.
     */
    template <std::size_t radius, std::size_t dim, class TInterval, class Func>
    void for_each_prediction_shift_run(const LevelCellArray<dim, TInterval>& domain,
                                       const std::array<typename TInterval::value_t, dim>& period,
                                       const TInterval& i,
                                       const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim - 1>>& index,
                                       Func&& f)
    {
        using value_t = typename TInterval::value_t;

        constexpr int r = static_cast<int>(radius);

        // Availability is only ever needed up to 2r: a stencil short by r on one side needs
        // 2r available opposite it, and nothing beyond that changes the answer. The
        // transverse rows the stencil can reach are the same 2r out, a shift of at most r
        // reading at most r further.
        constexpr int reach                = 2 * r;
        constexpr std::size_t rows_reached = detail::row_count<radius, dim>;

        const auto& order = detail::shift_search<radius, dim>();

        std::array<detail::RowScan<TInterval>, rows_reached> rows;
        for (std::size_t k = 0; k < rows_reached; ++k)
        {
            rows[k] = detail::wrapped_row_scan(domain, index, detail::nth_offset<dim - 1>(k, reach), period);
        }

        // The run being accumulated. Runs are emitted only when the shift changes, so two
        // neighbouring runs never carry the same one.
        bool pending = false;
        value_t run_start{};
        PredictionStencilShift<dim> run_shift;

        const auto flush = [&](value_t end)
        {
            if (pending)
            {
                f(TInterval{run_start, end, i.index}, run_shift);
            }
        };

        // How far each row reaches either side of x, capped at the reach, and where the
        // answer it gives stops holding.
        std::array<bool, rows_reached> covered{};
        std::array<int, rows_reached> low{};
        std::array<int, rows_reached> high{};

        value_t x = i.start;
        while (x < i.end)
        {
            value_t next = i.end;

            for (std::size_t k = 0; k < rows_reached; ++k)
            {
                value_t change = 0;
                covered[k]     = rows[k].covers(x, change);
                if (!covered[k])
                {
                    next = std::min(next, change);
                    continue;
                }

                const auto& run = rows[k].current();
                low[k]          = static_cast<int>(std::min(x - run.start, static_cast<value_t>(reach)));
                high[k]         = static_cast<int>(std::min(run.end - 1 - x, static_cast<value_t>(reach)));

                // The class changes cell by cell within 2r of each end of the run of cells
                // the row holds, and is constant in between - the bulk the consumers keep
                // their current kernel on. This reads the geometry, before the periodic
                // wrap below tops the counts back up: a wrap makes the two ends read the
                // same as the bulk, it does not make them one run with it.
                if (low[k] < reach || high[k] < reach)
                {
                    next = std::min(next, x + 1);
                }
                else
                {
                    next = std::min(next, run.end - static_cast<value_t>(reach));
                }

                // Off the end of a periodic direction the stencil reads the cells the
                // periodic exchange fills it from, one wrap away. One cell at a time,
                // because in x the count varies from cell to cell. A cell missing because
                // of a hole is not restored by this: its image one wrap away is outside the
                // domain, so the test fails.
                if (period[0] != 0)
                {
                    for (auto k_low = low[k]; k_low < reach && rows[k].contains(x - k_low - 1 + period[0]); ++k_low)
                    {
                        low[k] = k_low + 1;
                    }
                    for (auto k_high = high[k]; k_high < reach && rows[k].contains(x + k_high + 1 - period[0]); ++k_high)
                    {
                        high[k] = k_high + 1;
                    }
                }
            }

            // The most centred shift whose whole stencil box the domain holds. The x extent
            // is read off the rows the box covers: shifted by s, it reads [s - r, s + r]
            // around x, which each of those rows must hold.
            PredictionStencilShift<dim> shift;
            shift.fits = false;
            for (std::size_t c = 0; c < detail::shift_count<radius, dim> && !shift.fits; ++c)
            {
                bool admissible = true;
                for (std::size_t k = 0; k < detail::band_size<radius, dim> && admissible; ++k)
                {
                    const auto row = order.rows[c][k];
                    admissible     = covered[row] && order.shift[c][0] - r >= -low[row] && order.shift[c][0] + r <= high[row];
                }
                if (admissible)
                {
                    shift.shift = order.shift[c];
                    shift.fits  = true;
                }
            }

            // A breakpoint that failed to move would spin here rather than fail, so it is
            // asserted instead of being clamped away.
            assert(next > x && "for_each_prediction_shift_run: the run decomposition did not advance");

            if (!pending || !(shift == run_shift))
            {
                flush(x);
                pending   = true;
                run_start = x;
                run_shift = shift;
            }
            x = next;
        }

        flush(i.end);
    }

    /**
     * The prediction stencil shift at one cell. The per-interval form above is what the
     * hot kernels use; this is for consumers that already visit cells one at a time, and
     * for saying in one line what a test means.
     */
    template <std::size_t radius, std::size_t dim, class TInterval>
    PredictionStencilShift<dim> prediction_shifts_at(const LevelCellArray<dim, TInterval>& domain,
                                                     const std::array<typename TInterval::value_t, dim>& period,
                                                     const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim>>& coord)
    {
        using value_t = typename TInterval::value_t;

        xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> index;
        for (std::size_t d = 0; d + 1 < dim; ++d)
        {
            index[d] = coord[d + 1];
        }

        PredictionStencilShift<dim> shift;
        for_each_prediction_shift_run<radius>(domain,
                                              period,
                                              TInterval{coord[0], coord[0] + 1},
                                              index,
                                              [&](const auto&, const auto& run_shift)
                                              {
                                                  shift = run_shift;
                                              });
        return shift;
    }
}
