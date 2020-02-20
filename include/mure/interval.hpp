#pragma once

#include <ostream>

#include <spdlog/fmt/ostr.h>

namespace mure
{
    /** @class Interval
     *  @brief An interval with storage index.
     *
     * The index is used to associate each discrete coordinate @a c within
     * the interval to a value in a storage, at the position given by
     * @a index + @a c
     *
     * It implies that index may be negative but it allows to shrink
     * the interval (e.g. during a intersection operation) without the need
     * to recalculate the index.
     *
     * @tparam TValue  The coordinate type (must be signed).
     * @tparam TIndex  The index type (must be signed).
     */
    template<class TValue, class TIndex = signed long long int>
    struct Interval
    {
        static_assert(std::is_signed<TValue>::value,
                      "Coordinate type must be signed");
        static_assert(std::is_signed<TIndex>::value,
                      "Index type must be signed");

        using value_t = TValue;
        using index_t = TIndex;
        using coord_index_t = TValue;

        value_t start = 0; ///< Interval start.
        value_t end = 0;   ///< Interval end + 1.
        value_t step = 1;  ///< Step to move inside the Interval.
        index_t index = 0; ///< Storage index so that interval's content start
                           ///< at @a index + @a start.

        Interval() = default;
        Interval(Interval const &) = default;
        Interval(Interval &&) = default;
        Interval &operator=(Interval const &) = default;
        Interval &operator=(Interval &&) = default;

        inline Interval(value_t start, value_t end, index_t index = 0)
            : start{start}, end{end}, index{index}
        {}

        /// Returns true if the given coordinate lies within the interval.
        inline bool contains(value_t x) const
        {
            return (x >= start && x < end);
        }

        /// Returns the size (number of discrete coordinates) of the interval.
        inline auto size() const
        {
            return static_cast<std::size_t>(end - start);
        }

        /// Returns if the interval has a valid state (i.e. not empty).
        inline bool is_valid() const
        {
            return (start < end);
        }

        inline Interval<value_t, index_t> even_elements()
        {
            Interval<value_t, index_t> out{*this};

            out.start += (out.start&1)?1: 0;
            out.end -= (out.end&1)?0: 1;
            out.step = 2;
            return out;
        }

        inline Interval<value_t, index_t> odd_elements()
        {
            Interval<value_t, index_t> out{*this};

            out.start += (out.start&1)?0: 1;
            out.end -= (out.end&1)?1: 0;
            out.step = 2;
            return out;
        }

        inline Interval<value_t, index_t> &operator*=(value_t i)
        {
            start *= i;
            end *= i;
            step *= i;
            return *this;
        }

        inline Interval<value_t, index_t> &operator/=(value_t i)
        {
            start = std::floor(start/static_cast<double>(i));
            end = std::floor(end/static_cast<double>(i));
            if (start == end)
            {
                end++;
            }
            step = 1;
            return *this;
        }

        inline Interval<value_t, index_t> &operator>>=(std::size_t i)
        {
            bool add_one = (start == end) ? false : true;
            bool end_odd = (end & 1) ? true : false;
            start >>= i;
            end >>= i;
            if (end_odd or (start == end and add_one))
                end++;
            step = 1;
            return *this;
        }

        inline Interval<value_t, index_t> &operator<<=(std::size_t i)
        {
            start <<= i;
            end <<= i;
            step = 1;
            return *this;
        }

        inline Interval<value_t, index_t> &operator+=(value_t i)
        {
            start += i;
            end += i;
            return *this;
        }

        inline Interval<value_t, index_t> &operator-=(value_t i)
        {
            start -= i;
            end -= i;
            return *this;
        }
    };

    /// Display of an interval.
    template<class value_t, class index_t>
    inline std::ostream &operator<<(std::ostream &out,
                             const Interval<value_t, index_t> &interval)
    {
        out << "[" << interval.start << "," << interval.end << "[@"
            << interval.index << ":" << interval.step;
        return out;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator*(value_t i, const Interval<value_t, index_t> &interval)
    {
        auto that{interval};
        that *= i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator*(const Interval<value_t, index_t> &interval, value_t i)
    {
        auto that{interval};
        that *= i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator>>(const Interval<value_t, index_t> &interval, std::size_t i)
    {
        auto that{interval};
        that >>= i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator<<(const Interval<value_t, index_t> &interval, std::size_t i)
    {
        auto that{interval};
        that <<= i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator/(value_t i, const Interval<value_t, index_t> &interval)
    {
        auto that{interval};
        that /= i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator/(const Interval<value_t, index_t> &interval, value_t i)
    {
        auto that{interval};
        that /= i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator+(value_t i, const Interval<value_t, index_t> &interval)
    {
        auto that{interval};
        that += i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator+(const Interval<value_t, index_t> &interval, value_t i)
    {
        auto that{interval};
        that += i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator-(value_t i, const Interval<value_t, index_t> &interval)
    {
        auto that{interval};
        that -= i;
        return that;
    }

    template<class value_t, class index_t>
    inline Interval<value_t, index_t>
    operator-(const Interval<value_t, index_t> &interval, value_t i)
    {
        auto that{interval};
        that -= i;
        return that;
    }

    template<class value_t, class index_t>
    inline bool operator==(const Interval<value_t, index_t> &i1,
                    const Interval<value_t, index_t> &i2)
    {
        if (i1.start != i2.start or i1.end != i2.end or i1.step != i2.step or
            i1.index != i2.index)
            return false;
        return true;
    }

    template<class value_t, class index_t>
    inline bool operator!=(const Interval<value_t, index_t> &i1,
                    const Interval<value_t, index_t> &i2)
    {
        if (i1.start != i2.start or i1.end != i2.end or i1.step != i2.step or
            i1.index != i2.index)
            return true;
        return false;
    }
}
