// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>

// namespace samurai::experimental
namespace samurai
{
    template <class T>
    static constexpr T sentinel = std::numeric_limits<T>::max();

    template <class T>
    SAMURAI_INLINE T end_shift(T value, T shift)
    {
        return shift >= 0 ? value << shift : ((value - 1) >> -shift) + 1;
    }

    template <class T>
    SAMURAI_INLINE T start_shift(T value, T shift)
    {
        return shift >= 0 ? value << shift : value >> -shift;
    }

    constexpr auto compute_min(const auto& value, const auto&... args)
    {
        if constexpr (sizeof...(args) == 0u) // Single argument case!
        {
            return value;
        }
        else // For the Ts...
        {
            const auto min = compute_min(args...);
            return value < min ? value : min;
        }
    }

    constexpr auto compute_max(const auto& value, const auto&... args)
    {
        if constexpr (sizeof...(args) == 0u) // Single argument case!
        {
            return value;
        }
        else // For the Ts...
        {
            const auto max = compute_max(args...);
            return value > max ? value : max;
        }
    }

    template <class S1, class... S>
    struct get_interval_type
    {
        using type = typename S1::interval_t;
    };

    template <class... S>
    using get_interval_t = typename get_interval_type<S...>::type;

    template <class S1, class... S>
    struct get_set_dim
    {
        static constexpr std::size_t value = S1::dim;
    };

    template <class... S>
    constexpr std::size_t get_set_dim_v = get_set_dim<std::decay_t<S>...>::value;

    template <class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type::value_type>
    ForwardIt lower_bound_interval(ForwardIt begin, ForwardIt end, const T& value)
    {
        return std::lower_bound(begin,
                                end,
                                value,
                                [](const auto& interval, auto v)
                                {
                                    return interval.end <= v;
                                });
    }

    template <class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type::value_type>
    ForwardIt upper_bound_interval(ForwardIt begin, ForwardIt end, const T& value)
    {
        return std::upper_bound(begin,
                                end,
                                value,
                                [](auto v, const auto& interval)
                                {
                                    return v < interval.start;
                                });
    }

    template <typename Tuple1, typename Tuple2, typename F, std::size_t... Is>
    constexpr void zip_apply_impl(F&& f, Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...>)
    {
        (f(std::get<Is>(std::forward<Tuple1>(t1)), std::get<Is>(std::forward<Tuple2>(t2))), ...);
    }

    template <typename Tuple1, typename Tuple2, typename F>
    constexpr void zip_apply(F&& f, Tuple1&& t1, Tuple2&& t2)
    {
        constexpr std::size_t size1 = std::tuple_size_v<std::remove_reference_t<Tuple1>>;
        constexpr std::size_t size2 = std::tuple_size_v<std::remove_reference_t<Tuple2>>;

        static_assert(size1 == size2, "Tuples must have the same size");

        zip_apply_impl(std::forward<F>(f), std::forward<Tuple1>(t1), std::forward<Tuple2>(t2), std::make_index_sequence<size1>{});
    }
}
