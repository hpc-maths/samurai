// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"

#ifdef SAMURAI_MEASURE_SET_ALGEBRA
#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../timers.hpp"
#endif

namespace samurai
{
    namespace detail
    {
#ifdef SAMURAI_MEASURE_SET_ALGEBRA
        // Measurement build only (-DSAMURAI_MEASURE_SET_ALGEBRA=ON). When the
        // timers are enabled, apply() traverses the set expression twice: once
        // with a sink callback (timer "set algebra (pure)", cell column = rows,
        // i.e. the number of (y,z) lines whose x traverser was built) and once
        // for real (timer "set algebra + callback", cell column = intervals
        // emitted). The counters below are the run totals, printed by
        // samurai::finalize().
        struct SetAlgebraCounters
        {
            std::uint64_t applies   = 0;
            std::uint64_t rows      = 0;
            std::uint64_t intervals = 0;
            std::uint64_t cells     = 0;
            // consecutive rows (same z, y+1) emitting the same x intervals are one block
            std::uint64_t blocks = 0;
            // maximal groups of consecutive visited rows emitting nothing
            std::uint64_t empty_segments = 0;
        };

        // State of the block pass (see apply): which row is being visited and
        // whether it has emitted an interval yet.
        struct RowVisit
        {
            bool active                  = false; // a block pass is running
            bool row_open                = false;
            bool row_emitted             = false;
            bool in_empty_run            = false;
            std::uint64_t empty_segments = 0;
            std::array<long long, 3> yz{};
        };

        inline RowVisit& row_visit()
        {
            static thread_local RowVisit v;
            return v;
        }

        // Called at every (y,z) row entry of the x-dimension traversal.
        template <class YZ>
        inline void row_visit_enter(const YZ& yz, std::size_t n)
        {
            auto& v = row_visit();
            if (!v.active)
            {
                return;
            }
            if (v.row_open && !v.row_emitted)
            {
                // previous row was empty
                bool contiguous = (static_cast<long long>(yz[0]) == v.yz[0] + 1);
                for (std::size_t k = 1; k < n; ++k)
                {
                    contiguous = contiguous && static_cast<long long>(yz[k]) == v.yz[k];
                }
                if (!(v.in_empty_run && contiguous))
                {
                    ++v.empty_segments;
                }
                v.in_empty_run = true;
            }
            else if (v.row_open)
            {
                v.in_empty_run = false;
            }
            v.row_open    = true;
            v.row_emitted = false;
            for (std::size_t k = 0; k < n; ++k)
            {
                v.yz[k] = static_cast<long long>(yz[k]);
            }
        }

        inline void row_visit_end()
        {
            auto& v = row_visit();
            if (v.active && v.row_open && !v.row_emitted)
            {
                ++v.empty_segments; // the last visited row was empty (and not merged)
            }
            v.row_open = false;
        }

        inline SetAlgebraCounters& set_algebra_counters()
        {
            static SetAlgebraCounters counters;
            return counters;
        }

        // Same counters split by the innermost SAMURAI_MEASURE_PHASE scope.
        inline std::map<std::string, SetAlgebraCounters>& set_algebra_phase_counters()
        {
            static std::map<std::string, SetAlgebraCounters> counters;
            return counters;
        }

        inline std::uint64_t& set_algebra_row_counter()
        {
            static std::uint64_t rows = 0;
            return rows;
        }

        // SAMURAI_MEASURE_APPLY=0 in the environment disables the per-apply
        // passes (their timers cost ~1 us each), keeping only the phase timers.
        inline bool measure_apply_enabled()
        {
            static const bool enabled = []()
            {
                const char* v = std::getenv("SAMURAI_MEASURE_APPLY");
                return v == nullptr || std::string(v) != "0";
            }();
            return enabled;
        }
#endif

        template <class Set, class Func, std::size_t d>
        void apply_rec(const SetBase<Set>& set,
                       Func&& func,
                       typename SetBase<Set>::yz_index_t& yz_index,
                       std::integral_constant<std::size_t, d> d_ic,
                       typename SetBase<Set>::Workspace& workspace)
        {
            using traverser_t        = typename Set::template traverser_t<d>;
            using current_interval_t = typename traverser_t::current_interval_t;
            using interval_t         = typename traverser_t::interval_t;

            set.init_workspace(1, d_ic, workspace);

#ifdef SAMURAI_MEASURE_SET_ALGEBRA
            if constexpr (d == 0)
            {
                ++set_algebra_row_counter();
                if constexpr (Set::dim > 1)
                {
                    row_visit_enter(yz_index, Set::dim - 1);
                }
            }
#endif

            interval_t last_interval{-std::numeric_limits<typename interval_t::value_t>::max(),
                                     -std::numeric_limits<typename interval_t::value_t>::max() + 1};

            for (traverser_t traverser = set.get_traverser(yz_index, d_ic, workspace); !traverser.is_empty(); traverser.next_interval())
            {
                current_interval_t interval = traverser.current_interval();

                assert(last_interval < interval);

                if constexpr (d == 0)
                {
                    func(interval, yz_index);
                }
                else
                {
                    for (yz_index[d - 1] = interval.start; yz_index[d - 1] != interval.end; ++yz_index[d - 1])
                    {
                        apply_rec(set, std::forward<Func>(func), yz_index, std::integral_constant<std::size_t, d - 1>{}, workspace);
                    }
                }
            }
        }
    }

    template <class Set, class Func>
    void apply(const SetBase<Set>& set, Func&& func)
    {
        using Workspace  = typename Set::Workspace;
        using yz_index_t = typename Set::yz_index_t;

        constexpr std::size_t dim = Set::dim;

        yz_index_t yz_index;
        yz_index.fill(0); // to prevent -Wmaybe-uninitialized

        if (set.exist())
        {
#ifdef SAMURAI_MEASURE_SET_ALGEBRA
            if (times::timers.is_enabled() && detail::measure_apply_enabled())
            {
                auto& counters              = detail::set_algebra_counters();
                std::uint64_t& row_counter  = detail::set_algebra_row_counter();
                const std::uint64_t rows_at = row_counter;
                std::uint64_t intervals     = 0;
                std::uint64_t cells         = 0;

                times::timers.start("set algebra (pure)");
                {
                    Workspace sink_workspace;
                    auto sink = [&intervals, &cells](const auto& interval, const auto&)
                    {
                        ++intervals;
                        cells += static_cast<std::uint64_t>(interval.size());
                    };
                    detail::apply_rec(set, sink, yz_index, std::integral_constant<std::size_t, dim - 1>{}, sink_workspace);
                    // keep the traversal alive: its only visible effect is these two sums
                    asm volatile("" : : "r"(intervals), "r"(cells) : "memory");
                }
                const std::uint64_t rows = detail::set_algebra_row_counter() - rows_at;
                times::timers.stop("set algebra (pure)", rows);

                // Row blocks: a third, untimed pass groups consecutive rows (same z,
                // y + 1) that emit the same x intervals. This is the number of
                // callbacks a block-granularity traversal would make at best.
                std::uint64_t blocks         = 0;
                std::uint64_t empty_segments = 0;
                if constexpr (dim > 1)
                {
                    using value_t = typename Set::value_t;

                    struct Row
                    {
                        std::vector<std::pair<value_t, value_t>> xs;
                        yz_index_t yz;
                        bool valid = false;
                    };

                    static thread_local Row prev, cur;
                    prev.valid = false;
                    cur.valid  = false;
                    cur.xs.clear();
                    auto flush = [&]()
                    {
                        if (!cur.valid)
                        {
                            return;
                        }
                        bool same = prev.valid && prev.xs == cur.xs && cur.yz[0] == prev.yz[0] + 1;
                        for (std::size_t k = 1; same && k < dim - 1; ++k)
                        {
                            same = cur.yz[k] == prev.yz[k];
                        }
                        if (!same)
                        {
                            ++blocks;
                        }
                        std::swap(prev, cur);
                        prev.valid = true;
                        cur.valid  = false;
                        cur.xs.clear();
                    };
                    yz_index_t yz_blocks;
                    yz_blocks.fill(0);
                    Workspace block_workspace;
                    auto& visit     = detail::row_visit();
                    visit           = detail::RowVisit{};
                    visit.active    = true;
                    auto block_sink = [&](const auto& interval, const auto& yz)
                    {
                        visit.row_emitted = true;
                        bool same_row     = cur.valid;
                        for (std::size_t k = 0; same_row && k < dim - 1; ++k)
                        {
                            same_row = cur.yz[k] == yz[k];
                        }
                        if (cur.valid && !same_row)
                        {
                            flush();
                        }
                        cur.valid = true;
                        cur.yz    = yz;
                        cur.xs.emplace_back(interval.start, interval.end);
                    };
                    detail::apply_rec(set, block_sink, yz_blocks, std::integral_constant<std::size_t, dim - 1>{}, block_workspace);
                    flush();
                    detail::row_visit_end();
                    visit.active   = false;
                    empty_segments = visit.empty_segments;
                    row_counter    = rows_at + rows; // this pass is not part of the row count
                }
                else
                {
                    blocks = intervals;
                }

                yz_index.fill(0);
                times::timers.start("set algebra + callback");
                {
                    Workspace workspace;
                    detail::apply_rec(set, std::forward<Func>(func), yz_index, std::integral_constant<std::size_t, dim - 1>{}, workspace);
                }
                times::timers.stop("set algebra + callback", intervals);

                auto accumulate = [&](detail::SetAlgebraCounters& c)
                {
                    ++c.applies;
                    c.rows += rows;
                    c.intervals += intervals;
                    c.cells += cells;
                    c.blocks += blocks;
                    c.empty_segments += empty_segments;
                };
                accumulate(counters);
                accumulate(detail::set_algebra_phase_counters()[measure::current_phase()]);
                return;
            }
#endif
            Workspace workspace;
            detail::apply_rec(set, std::forward<Func>(func), yz_index, std::integral_constant<std::size_t, dim - 1>{}, workspace);
        }
    }
}
