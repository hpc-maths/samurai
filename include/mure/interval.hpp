#pragma once

#include <algorithm>
#include <iostream>
#include <list>
#include <forward_list>
#include <cstddef>

#ifdef USE_FAST_POOL
#include <boost/pool/pool_alloc.hpp>
#endif

namespace mure
{
    /** An interval with storage index.
     *
     * @tparam TValue   Coordonate type (must be signed)
     * @tparam TIndex   Index type
     */
    template<typename TValue, typename TIndex = std::size_t>
    struct Interval
    {
        static_assert(std::is_signed<TValue>::value, "Coordinate type must be signed");

        using value_t = TValue;
        using index_t = TIndex;

        value_t start = 0;  ///< Interval start
        value_t end   = 0;  ///< Interval end + 1
        index_t index = 0;  ///< Storage index where start the interval's content.

        Interval() = default;

        Interval(value_t start, value_t end, index_t index = 0): start{start}, end{end}, index{index}
        {}

        inline bool contains(value_t x) const
        {
            return (x >= start && x < end);
        }

        inline value_t size() const
        {
            return (end - start);
        }

        inline bool is_valid() const
        {
            return (start < end);
        }
    };

#ifdef USE_FAST_POOL
    template <typename T>
    using Allocator = boost::fast_pool_allocator<
        T,
        boost::default_user_allocator_new_delete,
        boost::details::pool::default_mutex
    >;
#endif

#ifdef USE_FAST_POOL
    template<typename TValue, typename TIndex = std::size_t>
    struct ListOfIntervals: private std::forward_list<Interval<TValue, TIndex>, Allocator<Interval<TValue, TIndex>>>
#else
    template<typename TValue, typename TIndex = std::size_t>
    struct ListOfIntervals: private std::forward_list<Interval<TValue, TIndex>>
#endif
    {
        using value_t       = TValue;
        using index_t       = TIndex;
        using interval_t    = Interval<value_t, index_t>;

#ifdef USE_FAST_POOL
        using list_t = std::forward_list<Interval<TValue, TIndex>, Allocator<Interval<TValue, TIndex>>>;
#else
        using list_t = std::forward_list<interval_t>;
#endif

        using typename list_t::forward_list;
        using list_t::begin;
        using list_t::end;

        using typename list_t::iterator;
        using typename list_t::const_iterator;
        using typename list_t::value_type;

        // Divides performances by two if you uncomment following lines (even without updating m_size)
        /*
        std::size_t m_size = 0;

        std::size_t size() const
        {
            return m_size;
        }
        */

        void add_interval(interval_t const& interval)
        {
            auto predicate = [&interval](auto const& value){return interval.start <= value.end;};
            auto it_last = this->before_begin();
            auto it = this->begin();
            while (it != end() && ! predicate(*it))
            {
                it_last = it++;
            }

            // if we are at the end just append the new interval or
            // if we are between two intervals, insert it
            if (it == end() || interval.end < (*it).start)
            {
                this->insert_after(it_last, interval);
                //++m_size;
                return;
            }

            // else there is an overlap
            (*it).start = std::min((*it).start, interval.start);
            (*it).end = std::max((*it).end, interval.end);

            auto it_end = std::next(it);
            while (it_end != end() && interval.end >= (*it_end).start)
            {
                 (*it).end = std::max((*it_end).end, interval.end);
                 it_end = this->erase_after(it);
                 //--m_size;
            }
        }
    };

    template<typename value_t, typename index_t>
    std::ostream& operator<<(std::ostream& out, const Interval<value_t, index_t>& interval)
    {
        out << "[" << interval.start << "," << interval.end << "[@" << interval.index;
        return out;
    }

    template<typename value_t, typename index_t>
    std::ostream& operator<<(std::ostream& out, ListOfIntervals<value_t, index_t> interval_list)
    {
        for(auto &interval: interval_list)
            out << interval << " ";
        return out;
    }
}
