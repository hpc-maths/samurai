// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

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
 * needs. Per-direction availability cannot see a cell that is missing only diagonally:
 *
 *         # # # . .        # a cell of the domain, . a cell of a hole
 *         # # # . .
 *         # # c # #        every cell c reads along an axis is there, yet the
 *         # # # # #        top-right corner of the radius-1 box around c is in
 *                          the hole - only the box rule sees it and shifts
 *
 * The two rules agree everywhere else - on a box domain they agree at every cell, corners
 * included - so this is a statement about re-entrant corners only, whether they belong to a
 * hole or to an L-shaped domain.
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
 * classifying the whole interval by its worst cell would move interior values:
 *
 *         the interval:      # # # # # # # # # # # #
 *         a hole below it:           . . . .
 *         its shift along y: 0 0 0 ^ ^ ^ ^ ^ ^ 0 0 0     ^ shifted away from the
 *                                                          hole, one cell past it
 *                                                          each side at radius 1
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

    /**
     * The shift to use along @a d, given what the query concluded.
     *
     * Where no shift fits, the centred stencil is kept. Two situations reach that, and both
     * are the behaviour of today's code rather than a new one:
     *   - the cell is outside the domain, where prediction reads the outer ghosts the
     *     boundary conditions write and there is nothing to shift into;
     *   - the band available at that level is too narrow to hold the stencil however it is
     *     shifted, which is the constructibility condition the mesh must guarantee. Saying
     *     so loudly rather than carrying on is a diagnostic of its own, not this function's
     *     business.
     */
    template <std::size_t dim>
    constexpr int shift_of(const PredictionStencilShift<dim>& shifts, std::size_t d)
    {
        return shifts.fits ? shifts.shift[d] : 0;
    }

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
         * For the transverse row tables, @ref TransverseRows::index_of is its inverse.
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
         * only stepping off the *end* of the domain wraps, and a hole's image one wrap
         * away is outside the domain. Only a direction the displacement stepped in, and
         * that is periodic, can have stepped off the end; when two of them have at once -
         * the corner of a doubly periodic 3D domain - wrapping one of them alone leaves
         * the row still empty, so the combinations are tried fewest wraps first and the
         * first non-empty one wins.
         */
        template <std::size_t dim, class TInterval>
        RowScan<TInterval> displaced_row(const LevelCellArray<dim, TInterval>& domain,
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

            auto scan = row_scan(domain, base);
            if (!scan.empty())
            {
                return scan;
            }

            // The directions that can have stepped off the end.
            std::array<std::size_t, cross> wrappable{};
            std::size_t n = 0;
            for (std::size_t d = 0; d < cross; ++d)
            {
                if (offset[d] != 0 && period[d + 1] != 0)
                {
                    wrappable[n++] = d;
                }
            }

            // Each subset of them, as a mask over the wrappable list.
            for (int wraps = 1; wraps <= static_cast<int>(n); ++wraps)
            {
                for (std::size_t mask = 1; mask < (std::size_t{1} << n); ++mask)
                {
                    if (std::popcount(mask) != wraps)
                    {
                        continue;
                    }

                    auto wrapped = base;
                    for (std::size_t b = 0; b < n; ++b)
                    {
                        if ((mask >> b) & std::size_t{1})
                        {
                            const auto d = wrappable[b];
                            wrapped[d] -= static_cast<value_t>(offset[d] > 0 ? 1 : -1) * period[d + 1];
                        }
                    }

                    auto candidate = row_scan(domain, wrapped);
                    if (!candidate.empty())
                    {
                        return candidate;
                    }
                }
            }
            return scan;
        }

        /**
         * How far one row covers around a cell:
         *
         *                            x
         *             [ = = = = = = # = = = = = = = = = ]    the run the row holds x in
         *               <-- low --> ^ <----- high ----->     both capped at the reach
         *
         * @c holds is false when the row does not hold the cell itself, and the counts
         * include the cells a periodic wrap reads off either end of the row. @c until is
         * the first coordinate above the cell at which any of this may change, so a
         * caller sweeping a range knows how far the answer it just got remains valid.
         */
        template <class value_t>
        struct RowCover
        {
            bool holds    = false;
            int low       = 0;
            int high      = 0;
            value_t until = 0;
        };

        /**
         * One row of the periodically extended domain. Which row of the domain that is
         * was settled by @ref displaced_row - the transverse half of the wrap; this class
         * adds the wrap *along* the row, so that off either end of a periodic direction
         * the cover keeps counting through the cells the periodic ghost exchange fills
         * the stencil from.
         */
        template <class TInterval>
        class DomainRow
        {
          public:

            using value_t = typename TInterval::value_t;

            DomainRow() = default;

            DomainRow(const RowScan<TInterval>& scan, value_t wrap, int reach)
                : m_scan(scan)
                , m_wrap(wrap)
                , m_reach(reach)
            {
            }

            /**
             * The cover around @a x - queried at increasing @a x, as @ref RowScan
             * requires.
             *
             * @c until is read off the geometry alone, *before* the wrap below tops the
             * counts back up: within the reach of either end of a run the cover changes
             * cell by cell, and in between it is constant - the bulk, where the consumers
             * keep their current kernel:
             *
             *        [ + + + + | = = = = = = = = = = = | + + + + ]
             *          2r cells:   the bulk: constant    2r cells:
             *          changes     until run.end - 2r    changes
             *          each cell                         each cell
             *
             * A wrap makes the two ends read the same as the bulk; it does not make them
             * one run with it.
             */
            RowCover<value_t> around(value_t x)
            {
                RowCover<value_t> cover;
                cover.holds = m_scan.covers(x, cover.until);
                if (!cover.holds)
                {
                    return cover;
                }

                const auto& run = m_scan.current();
                cover.low       = static_cast<int>(std::min(x - run.start, static_cast<value_t>(m_reach)));
                cover.high      = static_cast<int>(std::min(run.end - 1 - x, static_cast<value_t>(m_reach)));
                cover.until     = (cover.low < m_reach || cover.high < m_reach) ? x + 1 : run.end - static_cast<value_t>(m_reach);

                // Off the end of a periodic direction the stencil reads the cells the
                // periodic exchange fills it from, one wrap away. One cell at a time,
                // because along the row the count varies from cell to cell. A cell
                // missing because of a hole is not restored by this: its image one wrap
                // away is outside the domain, so the test fails.
                if (m_wrap != 0)
                {
                    while (cover.low < m_reach && m_scan.contains(x - cover.low - 1 + m_wrap))
                    {
                        ++cover.low;
                    }
                    while (cover.high < m_reach && m_scan.contains(x + cover.high + 1 - m_wrap))
                    {
                        ++cover.high;
                    }
                }
                return cover;
            }

          private:

            RowScan<TInterval> m_scan;
            value_t m_wrap = 0;
            int m_reach    = 0;
        };

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

        /**
         * The transverse rows within @c reach_ of a cell. For the shift query the reach is
         * `2r`: availability is only ever needed up to `2r` from the cell, since a stencil
         * short by r on one side needs 2r available opposite it, and nothing beyond that
         * changes the answer. The same 2r bounds the transverse reach, a shift of at most r
         * reading at most r further out:
         *
         *        -2r        -r         0         +r        +2r
         *         [==========(=========c=========)==========]
         *                     the centred stencil            [ ] everything any
         *                                                        shift can reach
         *
         * The type owns both the enumeration of the `[-reach, reach]^(dim-1)` transverse
         * offsets and the index of an offset in that enumeration, so an index computed with
         * @ref index_of points into a row array built with @ref offset by construction
         * rather than by convention.
         */
        template <int reach_, std::size_t dim>
        struct TransverseRows
        {
            static constexpr int reach         = reach_;
            static constexpr std::size_t width = 2 * static_cast<std::size_t>(reach) + 1;
            static constexpr std::size_t count = ipow(width, dim - 1);

            /// The @a k-th transverse offset, `k < count`.
            static constexpr std::array<int, dim - 1> offset(std::size_t k)
            {
                return nth_offset<dim - 1>(k, reach);
            }

            /// Where a transverse offset sits in the enumeration - @ref offset 's inverse.
            static constexpr std::size_t index_of(const std::array<int, dim - 1>& o)
            {
                std::size_t flat   = 0;
                std::size_t stride = 1;
                for (std::size_t d = 0; d + 1 < dim; ++d)
                {
                    flat += static_cast<std::size_t>(o[d] + reach) * stride;
                    stride *= width;
                }
                return flat;
            }
        };

        /**
         * The candidate shifts in the order they are preferred: `[-r, r]^dim`, sorted by
         * @ref more_centred. It depends on nothing but @c radius and @c dim, so it is built
         * once, and taking the first admissible entry is the whole selection rule.
         *
         * On the heap, and holding the candidates only. The rows each candidate reads were
         * tabulated too, as `count x band` indices, and that table is `(2r+1)^(2 dim - 1)`
         * entries: 16 GB of static storage at radius 3 in six dimensions, which is what the
         * prediction roundtrip test instantiates. The rows are cheap to recompute where they
         * are needed, so they are.
         */
        template <std::size_t radius, std::size_t dim>
        struct ShiftSearch
        {
            /// The shifts a stencil can take: `[-r, r]^dim`.
            static constexpr std::size_t count = ipow(2 * radius + 1, dim);
            /// The rows one stencil box covers: `[-r, r]^(dim-1)` around its shift.
            static constexpr std::size_t band = ipow(2 * radius + 1, dim - 1);
        };

        template <std::size_t radius, std::size_t dim>
        const std::vector<std::array<int, dim>>& shift_search()
        {
            constexpr int r = static_cast<int>(radius);
            using search_t  = ShiftSearch<radius, dim>;

            static const std::vector<std::array<int, dim>> order = []
            {
                std::vector<std::array<int, dim>> out(search_t::count);
                for (std::size_t c = 0; c < search_t::count; ++c)
                {
                    out[c] = nth_offset<dim>(c, r);
                }
                std::sort(out.begin(), out.end(), more_centred<dim>);
                return out;
            }();

            return order;
        }

        /**
         * The most centred shift whose whole stencil box the domain holds, read off the
         * cover of every row the stencil can reach: shifted by @c s, the box reads
         * `[s_0 - r, s_0 + r]` along each of the rows it covers, so @c s is admissible
         * exactly when each of those rows holds the cell and covers at least that far.
         *
         * @param covers one cover per row of `TransverseRows<2r, dim>`, in that enumeration
         */
        template <std::size_t radius, std::size_t dim, class value_t>
        PredictionStencilShift<dim> most_centred_fit(const RowCover<value_t>* covers)
        {
            constexpr int r             = static_cast<int>(radius);
            constexpr std::size_t cross = dim - 1;
            using search_t              = ShiftSearch<radius, dim>;
            using rows_t                = TransverseRows<2 * r, dim>;
            const auto& order           = shift_search<radius, dim>();

            PredictionStencilShift<dim> best;
            best.fits = false;
            for (std::size_t c = 0; c < search_t::count && !best.fits; ++c)
            {
                const auto& shift = order[c];
                bool admissible   = true;
                for (std::size_t k = 0; k < search_t::band && admissible; ++k)
                {
                    const auto within = nth_offset<cross>(k, r);
                    std::array<int, cross> transverse{};
                    for (std::size_t d = 0; d < cross; ++d)
                    {
                        transverse[d] = shift[d + 1] + within[d];
                    }
                    const auto& cover = covers[rows_t::index_of(transverse)];
                    admissible        = cover.holds && shift[0] - r >= -cover.low && shift[0] + r <= cover.high;
                }
                if (admissible)
                {
                    best.shift = shift;
                    best.fits  = true;
                }
            }
            return best;
        }

        /**
         * A fixed-size buffer for one query: on the stack while it is small, on the heap
         * once it is not. The row arrays are `(4r+1)^(dim-1)` entries, five in 2D at radius
         * 1 and 371293 at radius 3 in six dimensions, and the second is not a stack object.
         */
        template <class T, std::size_t count>
        class QueryScratch
        {
          public:

            static constexpr bool on_stack = count * sizeof(T) <= 4096;

            QueryScratch()
            {
                if constexpr (!on_stack)
                {
                    m_data.resize(count);
                }
            }

            T& operator[](std::size_t k)
            {
                return m_data[k];
            }

            const T* data() const
            {
                return m_data.data();
            }

          private:

            std::conditional_t<on_stack, std::array<T, count>, std::vector<T>> m_data{};
        };
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
        using rows_t  = detail::TransverseRows<2 * static_cast<int>(radius), dim>;

        // One cursor per transverse row the stencil can reach, over the periodically
        // extended domain.
        detail::QueryScratch<detail::DomainRow<TInterval>, rows_t::count> rows;
        for (std::size_t k = 0; k < rows_t::count; ++k)
        {
            rows[k] = detail::DomainRow<TInterval>(detail::displaced_row(domain, index, rows_t::offset(k), period), period[0], rows_t::reach);
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

        detail::QueryScratch<detail::RowCover<value_t>, rows_t::count> covers;

        value_t x = i.start;
        while (x < i.end)
        {
            // The cover of every row around x, and the first coordinate at which any of
            // those covers may change - before which the shift cannot change either.
            value_t next = i.end;
            for (std::size_t k = 0; k < rows_t::count; ++k)
            {
                covers[k] = rows[k].around(x);
                next      = std::min(next, covers[k].until);
            }

            const auto shift = detail::most_centred_fit<radius, dim>(covers.data());

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

    /**
     * The periodic wrap of each direction of @a mesh, seen at @a level, in the form
     * @ref for_each_prediction_shift_run takes: @c 0 where the direction is not periodic.
     *
     * Computed exactly as @c update_ghost_periodic computes the shift it copies by, so that
     * what prediction believes about a periodic direction and what fills the ghosts there
     * cannot drift apart.
     *
     * The mesh caches its domain's bounding box, so this is a handful of integer operations
     * and returns immediately when no direction is periodic - it is called per interval.
     */
    template <class Mesh>
    auto prediction_period(const Mesh& mesh, std::size_t level)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using value_t                    = typename Mesh::interval_t::value_t;

        const auto& domain = mesh.domain();
        assert(level <= domain.level() && "prediction_period: no periodic wrap is defined below the domain's own level");
        const auto delta_l = domain.level() - level;

        std::array<value_t, dim> period{};
        if (!mesh.is_periodic())
        {
            return period;
        }

        const auto& bbox = mesh.domain_bbox();
        for (std::size_t d = 0; d < dim; ++d)
        {
            period[d] = mesh.is_periodic(d) ? static_cast<value_t>((bbox[d].second - bbox[d].first) >> delta_l) : 0;
        }
        return period;
    }

    /**
     * The domain at one level, as the shift query reads it: the cells, the periodic wrap of
     * each direction, and - when the cells are exactly their bounding box - that box.
     *
     * On a box the position of a stencil is arithmetic on the box: the availability along a
     * direction is the distance to the box's two ends, or the reach when the direction is
     * periodic, and no row of the domain has to be scanned. That is what keeps the query out
     * of the profile on the common case, and what makes it affordable at all in many
     * dimensions, where the rows a stencil can reach number `(4r+1)^(dim-1)`. The two paths
     * are the same rule - on a box, availability per direction and the joint box test agree
     * at every cell - and tests/test_prediction_domain.cpp holds them to that cell for cell.
     *
     * Built by @ref prediction_domain from a mesh, which caches both the box and whether the
     * level is one.
     */
    template <std::size_t dim, class TInterval>
    struct PredictionDomain
    {
        using value_t = typename TInterval::value_t;
        using box_t   = std::array<std::pair<value_t, value_t>, dim>;

        PredictionDomain(const LevelCellArray<dim, TInterval>& cells_,
                         const std::array<value_t, dim>& period_,
                         bool is_box_,
                         const box_t& box_,
                         bool clamp_ = true)
            : cells(cells_)
            , period(period_)
            , is_box(is_box_)
            , box(box_)
            , clamp(clamp_)
        {
        }

        const LevelCellArray<dim, TInterval>& cells; ///< the domain at the level
        std::array<value_t, dim> period{};           ///< the periodic wrap per direction, 0 where the direction is not periodic
        bool is_box = false;                         ///< the cells are exactly their bounding box
        box_t box{};                                 ///< that bounding box, `[first, second)` per direction; read only when @c is_box
        bool clamp = true;                           ///< whether a stencil may shift inward at all (see @ref holds_prediction_inward_reach)
    };

    /**
     * Whether a mesh type guarantees that every cell a clamped prediction stencil reads is held
     * *and filled* at the level it is read from - the inward reach of `2r` cells that MRMesh's
     * update_sub_mesh_impl builds and its ghost update fills. A mesh declares it with
     * `static constexpr bool holds_inward_prediction_reach = true;`.
     *
     * Without that guarantee, shifting a stencil inward reads cells the mesh may hold but never
     * writes - the ghosts of a coarse level under a finer region, two cells in from a boundary,
     * which an AMR mesh with a one-cell ghost layer holds because they neighbour a coarse cell
     * yet cannot project, their children not being held - and the values it predicts from them
     * are whatever the allocation left there. Such a mesh keeps the centred stencil everywhere
     * and reads its outer ghosts, as it always did.
     */
    template <class Mesh, class = void>
    struct holds_prediction_inward_reach : std::false_type
    {
    };

    template <class Mesh>
    struct holds_prediction_inward_reach<Mesh, std::void_t<decltype(Mesh::holds_inward_prediction_reach)>>
        : std::bool_constant<Mesh::holds_inward_prediction_reach>
    {
    };

    /// The domain of @a mesh at @a level, in the form the shift query reads.
    template <class Mesh>
    auto prediction_domain(const Mesh& mesh, std::size_t level)
    {
        using domain_t = PredictionDomain<Mesh::dim, typename Mesh::interval_t>;
        return domain_t{mesh.domain(level),
                        prediction_period(mesh, level),
                        mesh.domain_is_box(level),
                        mesh.domain_bbox(level),
                        holds_prediction_inward_reach<Mesh>::value};
    }

    namespace detail
    {
        /**
         * The run decomposition of @a i on a box domain, read off the box. Returns false when
         * the box cannot answer - a periodic direction narrower than the stencil, where the
         * wrap would read a cell twice - and the general query then takes over.
         *
         * Same contract as the general query: cells the domain does not hold report @c fits
         * false with a zero shift, whether or not the direction is periodic; along a
         * periodic direction the stencil is never clamped; and two adjacent runs never carry
         * the same shift.
         */
        template <std::size_t radius, std::size_t dim, class TInterval, class Func>
        bool box_shift_runs(const PredictionDomain<dim, TInterval>& domain,
                            const TInterval& i,
                            const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim - 1>>& index,
                            Func& f)
        {
            using value_t       = typename TInterval::value_t;
            constexpr int reach = 2 * static_cast<int>(radius);

            for (std::size_t d = 0; d < dim; ++d)
            {
                if (domain.period[d] != 0 && domain.box[d].second - domain.box[d].first < static_cast<value_t>(reach + 1))
                {
                    return false;
                }
            }

            const auto whole_interval_does_not_fit = [&]
            {
                f(TInterval{i.start, i.end, i.index}, PredictionStencilShift<dim>{{}, false});
                return true;
            };

            // The transverse directions are fixed over the interval: the row is either outside
            // the domain, or its shift along each of them is settled once.
            PredictionStencilShift<dim> transverse;
            for (std::size_t d = 1; d < dim; ++d)
            {
                const value_t x  = index[d - 1];
                const value_t lo = domain.box[d].first;
                const value_t hi = domain.box[d].second;
                if (x < lo || x >= hi)
                {
                    return whole_interval_does_not_fit();
                }
                if (domain.period[d] != 0)
                {
                    continue;
                }
                const auto s = prediction_shift(radius,
                                                std::min<int>(static_cast<int>(x - lo), reach),
                                                std::min<int>(static_cast<int>(hi - 1 - x), reach));
                if (!s.fits)
                {
                    return whole_interval_does_not_fit();
                }
                transverse.shift[d] = s.shift;
            }

            // Along x the shift changes cell by cell within reach of either end of the box and
            // is constant in between; outside the box nothing fits.
            const value_t lo = domain.box[0].first;
            const value_t hi = domain.box[0].second;

            bool pending = false;
            value_t run_start{};
            PredictionStencilShift<dim> run_shift;

            const auto segment = [&](value_t start, const PredictionStencilShift<dim>& shift)
            {
                if (pending && shift == run_shift)
                {
                    return;
                }
                if (pending)
                {
                    f(TInterval{run_start, start, i.index}, run_shift);
                }
                pending   = true;
                run_start = start;
                run_shift = shift;
            };

            value_t x = i.start;
            while (x < i.end)
            {
                if (x < lo)
                {
                    segment(x, PredictionStencilShift<dim>{{}, false});
                    x = std::min(i.end, lo);
                    continue;
                }
                if (x >= hi)
                {
                    segment(x, PredictionStencilShift<dim>{{}, false});
                    x = i.end;
                    continue;
                }

                auto shift   = transverse;
                value_t next = x + 1;
                if (domain.period[0] != 0)
                {
                    next = std::min(i.end, hi);
                }
                else
                {
                    const int low  = std::min<int>(static_cast<int>(x - lo), reach);
                    const int high = std::min<int>(static_cast<int>(hi - 1 - x), reach);
                    const auto s   = prediction_shift(radius, low, high);
                    if (!s.fits)
                    {
                        shift = PredictionStencilShift<dim>{{}, false};
                    }
                    else
                    {
                        shift.shift[0] = s.shift;
                    }
                    if (low == reach && high == reach)
                    {
                        next = std::min(i.end, hi - static_cast<value_t>(reach));
                    }
                }
                segment(x, shift);
                x = next;
            }

            if (pending)
            {
                f(TInterval{run_start, i.end, i.index}, run_shift);
            }
            return true;
        }
    }

    /**
     * @ref for_each_prediction_shift_run on a @ref PredictionDomain: the box arithmetic where
     * the level is a box, the general query otherwise. This is the form the consumers use.
     */
    template <std::size_t radius, std::size_t dim, class TInterval, class Func>
    void for_each_prediction_shift_run(const PredictionDomain<dim, TInterval>& domain,
                                       const TInterval& i,
                                       const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim - 1>>& index,
                                       Func&& f)
    {
        if (!domain.clamp)
        {
            // The mesh does not guarantee the inward reach a shifted stencil reads: centred
            // everywhere, reading the outer ghosts, as before the stencil could shift.
            f(i, PredictionStencilShift<dim>{});
            return;
        }
        if (domain.is_box && detail::box_shift_runs<radius>(domain, i, index, f))
        {
            return;
        }
        for_each_prediction_shift_run<radius>(domain.cells, domain.period, i, index, std::forward<Func>(f));
    }

    /// @ref prediction_shifts_at on a @ref PredictionDomain.
    template <std::size_t radius, std::size_t dim, class TInterval>
    PredictionStencilShift<dim> prediction_shifts_at(const PredictionDomain<dim, TInterval>& domain,
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
                                              TInterval{coord[0], coord[0] + 1},
                                              index,
                                              [&](const auto&, const auto& run_shift)
                                              {
                                                  shift = run_shift;
                                              });
        return shift;
    }
}
