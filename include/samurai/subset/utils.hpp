// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <limits>

// namespace samurai::experimental
namespace samurai
{
    template <class T>
    static constexpr T sentinel = std::numeric_limits<T>::max();

    template <class T>
    inline T end_shift(T value, T shift)
    {
        return shift >= 0 ? value << shift : ((value - 1) >> -shift) + 1;
    }

    template <class T>
    inline T start_shift(T value, T shift)
    {
        return shift >= 0 ? value << shift : value >> -shift;
    }

    constexpr auto compute_min(auto const& value, auto const&... args)
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

    constexpr auto compute_max(auto const& value, auto const&... args)
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
    constexpr std::size_t get_set_dim_v = get_set_dim<S...>::value;
}