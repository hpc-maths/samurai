// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <functional>
#include <tuple>

namespace samurai
{

    template <typename T, typename... Ts>
    T vmin(const T& a, const T& b, const Ts&... others)
    {
        if constexpr (sizeof...(Ts) == 0u)
        {
            return std::min(a, b);
        }
        else
        {
            return a < b ? vmin(a, others...) : vmin(b, others...);
        }
    }

    template <typename T, typename... Ts>
    T vmax(const T& a, const T& b, const Ts&... others)
    {
        if constexpr (sizeof...(Ts) == 0u)
        {
            return std::max(a, b);
        }
        else
        {
            return a > b ? vmax(a, others...) : vmax(b, others...);
        }
    }

    namespace detail
    {
        template <class Tuple, class Func, std::size_t... Is>
        Func enumerate_items(Tuple& tuple, Func func, std::index_sequence<Is...>)
        {
            (func(Is, std::get<Is>(tuple)), ...);
            return func;
        }

        template <class Tuple, class Func, std::size_t... Is>
        Func enumerate_const_items(const Tuple& tuple, Func func, std::index_sequence<Is...>)
        {
            (func(Is, std::get<Is>(tuple)), ...);
            return func;
        }
    }

    template <class Tuple, class Func>
    Func enumerate_items(Tuple& tuple, Func func)
    {
        constexpr std::size_t N = std::tuple_size_v<std::decay_t<Tuple>>;

        return detail::enumerate_items(tuple, func, std::make_index_sequence<N>{});
    }

    template <class Tuple, class Func>
    Func enumerate_const_items(const Tuple& tuple, Func func)
    {
        constexpr std::size_t N = std::tuple_size_v<std::decay_t<Tuple>>;

        return detail::enumerate_const_items(tuple, func, std::make_index_sequence<N>{});
    }
}
