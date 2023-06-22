// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <ostream>

#include <fmt/format.h>

#include "samurai_config.hpp"

namespace samurai
{

    /////////////////////////
    // Interval definition //
    /////////////////////////

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
    template <class TValue, class TIndex = default_config::index_t>
    struct Interval
    {
        static_assert(std::is_signed<TValue>::value, "Coordinate type must be signed");
        static_assert(std::is_signed<TIndex>::value, "Index type must be signed");

        using value_t       = TValue;
        using index_t       = TIndex;
        using coord_index_t = TValue;

        value_t start = 0; ///< Interval start.
        value_t end   = 0; ///< Interval end + 1.
        value_t step  = 1; ///< Step to move inside the Interval.
        index_t index = 0; ///< Storage index so that interval's content start
                           ///< at @a index + @a start.

        Interval() = default;
        Interval(value_t start, value_t end, index_t index = 0);

        bool contains(value_t x) const;
        std::size_t size() const;
        bool is_valid() const;

        Interval even_elements() const;
        Interval odd_elements() const;

        Interval& operator*=(value_t i);
        Interval& operator/=(value_t i);
        Interval& operator>>=(std::size_t i);
        Interval& operator<<=(std::size_t i);
        Interval& operator+=(value_t i);
        Interval& operator-=(value_t i);
    };

    /////////////////////////////
    // Interval implementation //
    /////////////////////////////

    template <class TValue, class TIndex>
    inline Interval<TValue, TIndex>::Interval(value_t start_, value_t end_, index_t index_)
        : start{start_}
        , end{end_}
        , index{index_}
    {
    }

    /**
     * Returns true if the given coordinate lies within the interval.
     */
    template <class TValue, class TIndex>
    inline bool Interval<TValue, TIndex>::contains(value_t x) const
    {
        return (x >= start && x < end);
    }

    /**
     * Returns the size (number of discrete coordinates) of the interval.
     */
    template <class TValue, class TIndex>
    inline std::size_t Interval<TValue, TIndex>::size() const
    {
        return static_cast<std::size_t>(end - start);
    }

    /**
     * Returns if the interval has a valid state (i.e. not empty).
     */
    template <class TValue, class TIndex>
    inline bool Interval<TValue, TIndex>::is_valid() const
    {
        return (start < end);
    }

    /**
     * Returns the even elements of the interval.
     *
     * @warning the result could be an invalid interval.
     */
    template <class TValue, class TIndex>
    inline auto Interval<TValue, TIndex>::even_elements() const -> Interval
    {
        Interval<value_t, index_t> out{*this};

        out.start += (out.start & 1) ? 1 : 0;
        out.end -= (out.end & 1) ? 0 : 1;
        out.step = 2;
        return out;
    }

    /**
     * Returns the odd elements of the interval.
     *
     * @warning the result could be an invalid interval.
     */
    template <class TValue, class TIndex>
    inline auto Interval<TValue, TIndex>::odd_elements() const -> Interval
    {
        Interval<value_t, index_t> out{*this};

        out.start += (out.start & 1) ? 0 : 1;
        out.end -= (out.end & 1) ? 1 : 0;
        out.step = 2;
        return out;
    }

    template <class TValue, class TIndex>
    inline auto Interval<TValue, TIndex>::operator*=(value_t i) -> Interval&
    {
        start *= i;
        end *= i;
        step *= i;
        return *this;
    }

    template <class TValue, class TIndex>
    inline auto Interval<TValue, TIndex>::operator/=(value_t i) -> Interval&
    {
        start = static_cast<value_t>(std::floor(start / static_cast<double>(i)));
        end   = static_cast<value_t>(std::floor(end / static_cast<double>(i)));
        if (start == end)
        {
            ++end;
        }
        step = 1;
        return *this;
    }

    template <class TValue, class TIndex>
    inline auto Interval<TValue, TIndex>::operator>>=(std::size_t i) -> Interval&
    {
        bool add_one = (start != end);
        bool end_odd = (end & 1);
        start >>= i;
        end >>= i;
        if (end_odd || (start == end && add_one))
        {
            ++end;
        }
        step = 1;
        return *this;
    }

    template <class TValue, class TIndex>
    inline auto Interval<TValue, TIndex>::operator<<=(std::size_t i) -> Interval&
    {
        start <<= i;
        end <<= i;
        step = 1;
        return *this;
    }

    template <class TValue, class TIndex>
    inline auto Interval<TValue, TIndex>::operator+=(value_t i) -> Interval&
    {
        start += i;
        end += i;
        return *this;
    }

    template <class TValue, class TIndex>
    inline auto Interval<TValue, TIndex>::operator-=(value_t i) -> Interval&
    {
        start -= i;
        end -= i;
        return *this;
    }

    /**
     * Display of an interval.
     */
    template <class value_t, class index_t>
    inline std::ostream& operator<<(std::ostream& out, const Interval<value_t, index_t>& interval)
    {
        out << "[" << interval.start << "," << interval.end << "[@" << interval.index << ":" << interval.step;
        return out;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator*(value_t i, const Interval<value_t, index_t>& interval)
    {
        auto that{interval};
        that *= i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator*(const Interval<value_t, index_t>& interval, value_t i)
    {
        auto that{interval};
        that *= i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator>>(const Interval<value_t, index_t>& interval, std::size_t i)
    {
        auto that{interval};
        that >>= i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator<<(const Interval<value_t, index_t>& interval, std::size_t i)
    {
        auto that{interval};
        that <<= i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator/(value_t i, const Interval<value_t, index_t>& interval)
    {
        auto that{interval};
        that /= i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator/(const Interval<value_t, index_t>& interval, value_t i)
    {
        auto that{interval};
        that /= i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator+(value_t i, const Interval<value_t, index_t>& interval)
    {
        auto that{interval};
        that += i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator+(const Interval<value_t, index_t>& interval, value_t i)
    {
        auto that{interval};
        that += i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator-(value_t i, const Interval<value_t, index_t>& interval)
    {
        auto that{interval};
        that -= i;
        return that;
    }

    template <class value_t, class index_t>
    inline Interval<value_t, index_t> operator-(const Interval<value_t, index_t>& interval, value_t i)
    {
        auto that{interval};
        that -= i;
        return that;
    }

    template <class value_t, class index_t>
    inline bool operator==(const Interval<value_t, index_t>& i1, const Interval<value_t, index_t>& i2)
    {
        return !(i1.start != i2.start || i1.end != i2.end || i1.step != i2.step || i1.index != i2.index);
    }

    template <class value_t, class index_t>
    inline bool operator!=(const Interval<value_t, index_t>& i1, const Interval<value_t, index_t>& i2)
    {
        return !(i1 == i2);
    }

    template <class value_t, class index_t>
    inline bool operator<(const Interval<value_t, index_t>& i1, const Interval<value_t, index_t>& i2)
    {
        return i1.start < i2.start;
    }
} // namespace samurai

template <class TValue, class TIndex>
struct fmt::formatter<samurai::Interval<TValue, TIndex>>
{
    constexpr auto parse(const format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const samurai::Interval<TValue, TIndex>& interval, FormatContext& ctx)
    {
        return format_to(ctx.out(), "[{}, {}[@{}:{}", interval.start, interval.end, interval.index, interval.step);
    }
};
