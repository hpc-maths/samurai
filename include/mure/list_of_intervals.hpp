#pragma once

#include <iostream>
#include <forward_list>

#include "interval.hpp"

namespace mure
{
    template<class InputIt, class UnaryPredicate>
    constexpr std::pair<InputIt, InputIt> forward_find_if(InputIt first, InputIt last, UnaryPredicate p)
    {
        auto previous = first++;
        for (; first != last; ++first, ++previous) {
            if (p(*first)) {
                break;
            }
        }
        return {previous, first};
    }

    template<typename TValue, typename TIndex = std::size_t>
    struct ListOfIntervals: private std::forward_list<Interval<TValue, TIndex>>
    {
        using value_t       = TValue;
        using index_t       = TIndex;
        using interval_t    = Interval<value_t, index_t>;

        using list_t = std::forward_list<interval_t>;
        using typename list_t::forward_list;
        using list_t::before_begin;
        using list_t::begin;
        using list_t::end;

        using list_t::erase_after;
        using typename list_t::iterator;
        using typename list_t::const_iterator;
        using typename list_t::value_type;

        std::size_t size() const
        {
            return std::distance(begin(), end());
        }

        void add_interval(interval_t&& interval)
        {
            auto predicate = [interval](auto const& value){return interval.start <= value.end;};
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
    std::ostream& operator<<(std::ostream& out, ListOfIntervals<value_t, index_t> interval_list)
    {
        for(auto &interval: interval_list)
            out << interval << " ";
        return out;
    }    
}

