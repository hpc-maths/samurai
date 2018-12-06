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

        Interval(value_t start, value_t end, index_t index=0): start{start}, end{end}, index{index}
        {}

        bool contains(value_t x) const
        {
            return x >= start && x < end;
        }

        value_t size() const
        {
            return end - start;
        }

        bool isvalid() const
        {
            return start < end;
        }
    };

    template<typename value_t, typename index_t>
    struct ListOfIntervals: private std::list<Interval<value_t, index_t>>
    {
        using interval_t = Interval<value_t, index_t>;
        using list_t = std::list<interval_t>;
        using list_t::list;
        using list_t::begin;
        using list_t::end;
        using list_t::insert;

        void add_interval(interval_t const& interval)
        {
            auto predicate = [&interval](auto const& value){return interval.start > value.end;};
            auto it = std::find_if(begin(), end(),predicate);

            // if we are at the end just append the new interval or
            // if we are between two intervals, insert it
            if (it == end() || interval.end < (*it).start)
            {
                insert(it, interval);
                return;
            }

            // # else there is an overlap
            // self[index].start = min(self[index].start, interval.start)
            // self[index].end = max(self[index].end, interval.end)

            // index_end = index + 1
            // while index_end < len(self.intervals) and interval.end >= self[index_end].start:
            //     self[index].end = max(self[index_end].end, interval.end)
            //     self.pop(index_end)
            // return
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