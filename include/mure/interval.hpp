#pragma once

#include<algorithm>
#include<iostream>
#include<list>

namespace mure
{
    template<typename value_t, typename index_t>
    struct Interval
    {
        value_t start;
        value_t end;
        index_t index = 0;

        Interval() = default;

        Interval(value_t start, value_t end, index_t index=0): start{start}, end{end}, index{index}
        {}

        inline bool contains(value_t x) const
        {
            return (x >= start && x < end);
        }

        inline value_t size() const
        {
            return (end - start);
        }

        inline bool isvalid() const
        {
            return (start < end);
        }
    };

    template<typename value_t, typename index_t>
    struct ListOfIntervals: private std::list<Interval<value_t, index_t>>
    {
        using interval_t = Interval<value_t, index_t>;
        using list_t = std::list<interval_t>;
        using std::list<interval_t>::list;
        using list_t::begin;
        using list_t::end;
        using list_t::insert;
        using list_t::erase;

        void add_interval(interval_t const& interval)
        {
            auto predicate = [&interval](auto const& value){return interval.start < value.end;};
            auto it = std::find_if(begin(), end(), predicate);

            // if we are at the end just append the new interval or
            // if we are between two intervals, insert it
            if (it == end() || interval.end < (*it).start)
            {
                insert(it, interval);
                return;
            }

            // else there is an overlap
            (*it).start = std::min((*it).start, interval.start);
            (*it).end = std::max((*it).end, interval.end);

            auto it_end = std::next(it);
            while (it_end != end() && interval.end >= (*it_end).start)
            {
                 (*it).end = std::max((*it_end).end, interval.end);
                 erase(it_end);
            }
        }
    };

    template<typename value_t, typename index_t>
    std::ostream& operator<<(std::ostream& out, const Interval<value_t, index_t>& interval)
    {
        out << "[" << interval.start << ", " << interval.end << ", index = " << interval.index << "]";
        return out;
    }

    template<typename value_t, typename index_t>
    std::ostream& operator<<(std::ostream& out, ListOfIntervals<value_t, index_t> interval_list)
    {
        for(auto &interval: interval_list)
            out << interval;
        return out;
    }
}