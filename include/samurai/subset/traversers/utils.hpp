// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <ranges>

#include "../utils.hpp"

namespace samurai
{

    namespace traverser_utils
    {

        template <typename T>
        SAMURAI_INLINE T refine_start(const T& interval_start, const std::size_t shift)
        {
            return interval_start << shift;
        }

        template <typename T>
        SAMURAI_INLINE T coarsen_start(const T& interval_start, const std::size_t shift)
        {
            return interval_start >> shift;
        }

        template <typename T>
        SAMURAI_INLINE T coarsen_end(const T& interval_end, const std::size_t shift)
        {
            return ((interval_end - 1) >> shift) + 1;
        }

        template <typename T>
        SAMURAI_INLINE T refine_end(const T& interval_end, const std::size_t shift)
        {
            return (interval_end << shift);
        }

        namespace detail
        {
            template <typename UnspecifiedSetTraversers>
            struct SetTraverserTupleInterval;

            template <typename... SetTraversers>
            struct SetTraverserTupleInterval<std::tuple<SetTraversers...>>
            {
                using Type = typename std::tuple_element_t<0, std::tuple<SetTraversers...>>::interval_t;
            };

            template <typename Range>
                requires(std::ranges::forward_range<Range> and IsSetTraverser<std::ranges::range_value_t<Range>>::value)
            struct SetTraverserTupleInterval<Range>
            {
                using Type = typename std::ranges::range_value_t<Range>::interval_t;
            };
        } // namespace detail

        template <typename SetTraversers>
        using SetTraverserTupleInterval = typename detail::SetTraverserTupleInterval<SetTraversers>::Type;

        template <typename SetTraversers, class StartFunc, class EndFunc>
        SAMURAI_INLINE auto transform_and_union(SetTraversers& set_traversers,
                                                const StartFunc startFunc,
                                                const EndFunc endFunc) -> SetTraverserTupleInterval<SetTraversers>
        {
            using interval_t = SetTraverserTupleInterval<SetTraversers>;
            using value_t    = typename interval_t::value_t;

            interval_t current_interval;

            current_interval.start = std::numeric_limits<value_t>::max();

            // We find the start of the interval, i.e. the smallest set_traverser.current_interval().start
            enumerate_items(
                set_traversers,
                [&](const std::size_t i, const auto& set_traverser) -> void
                {
                    if (!set_traverser.is_empty() && (startFunc(i, set_traverser.current_interval().start) < current_interval.start))
                    {
                        current_interval.start = startFunc(i, set_traverser.current_interval().start);
                        current_interval.end   = endFunc(i, set_traverser.current_interval().end);
                    }
                });
            // Now we find the end of the interval, i.e. the largest set_traverser.current_interval().end
            // such that set_traverser.current_interval().start - expansion < current_interval.end
            bool is_done = false;
            while (!is_done)
            {
                is_done = true;
                // advance set traverses that are behind current interval
                enumerate_items(
                    set_traversers,
                    [&](const std::size_t i, auto& set_traverser) -> void
                    {
                        while (!set_traverser.is_empty() && endFunc(i, set_traverser.current_interval().end) <= current_interval.end)
                        {
                            set_traverser.next_interval();
                        }
                    });
                // try to find a new end
                enumerate_items(
                    set_traversers,
                    [&](const std::size_t i, const auto& set_traverser) -> void
                    {
                        // there is an overlap
                        if (!set_traverser.is_empty() && startFunc(i, set_traverser.current_interval().start) <= current_interval.end)
                        {
                            is_done              = false;
                            current_interval.end = endFunc(i, set_traverser.current_interval().end);
                        }
                    });
            }

            return current_interval;
        }

    } // traverser_utils

} // samurai
