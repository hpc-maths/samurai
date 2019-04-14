#pragma once

#include <forward_list>
#include <iostream>

#include "interval.hpp"

namespace mure
{
    template<class InputIt, class UnaryPredicate>
    constexpr std::pair<InputIt, InputIt>
    forward_find_if(InputIt first, InputIt last, UnaryPredicate p)
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

    /** @class ListOfIntervals
     *  @brief Forward list of intervals.
     *
     * @tparam TValue  The coordinate type (must be signed).
     * @tparam TIndex  The index type (must be signed).
     */
    template<typename TValue, typename TIndex = signed long long int>
    struct ListOfIntervals : private std::forward_list<Interval<TValue, TIndex>>
    {
        using value_t = TValue;
        using index_t = TIndex;
        using interval_t = Interval<value_t, index_t>;

        using list_t = std::forward_list<interval_t>;
        using list_t::before_begin;
        using list_t::begin;
        using list_t::cbegin;
        using list_t::cend;
        using list_t::end;
        using typename list_t::forward_list;

        using list_t::erase_after;
        using typename list_t::const_iterator;
        using typename list_t::iterator;
        using typename list_t::value_type;

        /// Number of intervals stored in the list.
        std::size_t size() const
        {
            return static_cast<std::size_t>(std::distance(begin(), end()));
        }

        /// Add a point inside the list.
        void add_point(value_t point)
        {
            add_interval({point, point + 1});
        }

        /// Add an interval inside the list.
        void add_interval(interval_t &&interval)
        {
            auto predicate = [interval](auto const &value) {
                return interval.start <= value.end;
            };
            auto it = forward_find_if(before_begin(), end(), predicate);

            // if we are at the end just append the new interval or
            // if we are between two intervals, insert it
            if (it.second == end() || interval.end < it.second->start)
            {
                this->insert_after(it.first, std::move(interval));
                return;
            }

            // else there is an overlap
            it.second->start = std::min(it.second->start, interval.start);
            it.second->end = std::max(it.second->end, interval.end);

            auto it_end = std::next(it.second);
            while (it_end != end() && interval.end >= it_end->start)
            {
                it.second->end = std::max(it_end->end, interval.end);
                it_end = erase_after(it.second);
            }
        }
    };

    template<typename value_t, typename index_t>
    std::ostream &operator<<(std::ostream &out,
                             ListOfIntervals<value_t, index_t> interval_list)
    {
        for (auto &interval : interval_list)
            out << interval << " ";
        return out;
    }
}
