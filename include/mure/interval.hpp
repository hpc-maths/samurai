#pragma once

#include <ostream>

namespace mure
{
    /** An interval with storage index.
     *
     * The index is used to associate each discrete coordinate @a c within
     * the interval to a value in a storage, at the position given by
     * @a index + @a c
     *
     * It implies that index may be negative but it allows to shrink
     * the intervale (e.g. during a intersection operation) without the need
     * to recalculate the index.
     *
     * @tparam TValue   Coordinate type (must be signed)
     * @tparam TIndex   Index type (must be signed)
     */
    template <typename TValue, typename TIndex = signed long long int>
    struct Interval
    {
        static_assert(std::is_signed<TValue>::value, "Coordinate type must be signed");
        static_assert(std::is_signed<TIndex>::value, "Index type must be signed");

        using value_t = TValue;
        using index_t = TIndex;

        value_t start = 0;  ///< Interval start
        value_t end   = 0;  ///< Interval end + 1
        index_t index = 0;  ///< Storage index so that interval's content start at @a index + @a start

        Interval() = default;

        Interval(value_t start, value_t end, index_t index = 0)
            : start{start}, end{end}, index{index}
        {}

        /// Returns true if the given coordinate lies within the interval.
        inline bool contains(value_t x) const
        {
            return (x >= start && x < end);
        }

        /// Returns the size (number of discrete coordinates) of the interval.
        inline value_t size() const
        {
            return (end - start);
        }

        /// Returns if the interval has a valid state (i.e. not empty).
        inline bool is_valid() const
        {
            return (start < end);
        }
    };

    /// Display of an interval.
    template<typename value_t, typename index_t>
    std::ostream& operator<< (std::ostream& out, const Interval<value_t, index_t>& interval)
    {
        out << "[" << interval.start << "," << interval.end << "[@" << interval.index;
        return out;
    }
}
