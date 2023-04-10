// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <forward_list>
#include <iostream>

#include "interval.hpp"
#include "samurai_config.hpp"

namespace samurai
{
    namespace detail
    {
        template <class InputIt, class UnaryPredicate>
        inline constexpr std::pair<InputIt, InputIt> forward_find_if(InputIt first, InputIt last, UnaryPredicate p)
        {
            auto previous = first++;
            for (; first != last; ++first, ++previous)
            {
                if (p(*first))
                {
                    break;
                }
            }
            return {previous, first};
        }
    } // namespace detail

    ////////////////////////////////
    // ListOfIntervals definition //
    ////////////////////////////////

    /** @class ListOfIntervals
     *  @brief Forward list of intervals.
     *
     * @tparam TValue  The coordinate type (must be signed).
     * @tparam TIndex  The index type (must be signed).
     */
    template <typename TValue, typename TIndex = default_config::index_t>
    struct ListOfIntervals : private std::forward_list<Interval<TValue, TIndex>>
    {
        using value_t    = TValue;
        using index_t    = TIndex;
        using interval_t = Interval<value_t, index_t>;

        using list_t = std::forward_list<interval_t>;
        using list_t::before_begin;
        using list_t::begin;
        using list_t::cbegin;
        using list_t::cend;
        using list_t::empty;
        using list_t::end;

        using list_t::erase_after;
        using const_iterator = typename list_t::const_iterator;
        using iterator       = typename list_t::iterator;
        using value_type     = typename list_t::value_type;

        std::size_t size() const;

        void add_point(value_t point);
        void add_interval(const interval_t& interval);
    };

    ////////////////////////////////////
    // ListOfIntervals implementation //
    ////////////////////////////////////

    /// Number of intervals stored in the list.
    template <typename TValue, typename TIndex>
    inline std::size_t ListOfIntervals<TValue, TIndex>::size() const
    {
        return static_cast<std::size_t>(std::distance(begin(), end()));
    }

    /// Add a point inside the list.
    template <typename TValue, typename TIndex>
    inline void ListOfIntervals<TValue, TIndex>::add_point(value_t point)
    {
        add_interval({point, point + 1});
    }

    /// Add an interval inside the list.
    template <typename TValue, typename TIndex>
    inline void ListOfIntervals<TValue, TIndex>::add_interval(const interval_t& interval)
    {
        if (!interval.is_valid())
        {
            return;
        }

        auto predicate = [interval](const auto& value)
        {
            return interval.start <= value.end;
        };
        auto it = detail::forward_find_if(before_begin(), end(), predicate);

        // if we are at the end just append the new interval or
        // if we are between two intervals, insert it
        if (it.second == end() || interval.end < it.second->start)
        {
            this->insert_after(it.first, interval);
            return;
        }

        // else there is an overlap
        it.second->start = std::min(it.second->start, interval.start);
        it.second->end   = std::max(it.second->end, interval.end);

        auto it_end = std::next(it.second);
        while (it_end != end() && interval.end >= it_end->start)
        {
            it.second->end = std::max(it_end->end, interval.end);
            it_end         = erase_after(it.second);
        }
    }

    template <typename value_t, typename index_t>
    inline std::ostream& operator<<(std::ostream& out, const ListOfIntervals<value_t, index_t>& interval_list)
    {
        for (const auto& interval : interval_list)
        {
            out << interval << " ";
        }
        return out;
    }
} // namespace samurai
