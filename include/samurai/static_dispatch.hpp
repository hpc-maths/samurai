// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace samurai
{
    namespace detail
    {
        // Recursive binary dispatch: each leaf (min == max) does `return f(...)` directly,
        // so the selected branch's result is returned by value with guaranteed copy elision
        // (no intermediate staging), for both void and non-void callables.
        template <std::size_t min, std::size_t max, class F>
        constexpr decltype(auto) dispatch_static_impl(std::size_t value, F&& f)
        {
            if constexpr (min == max)
            {
                if (value != min)
                {
                    throw std::out_of_range("dispatch_static: value out of range");
                }
                return std::forward<F>(f)(std::integral_constant<std::size_t, min>{});
            }
            else
            {
                constexpr std::size_t mid = min + (max - min) / 2;
                if (value <= mid)
                {
                    return dispatch_static_impl<min, mid>(value, std::forward<F>(f));
                }
                else
                {
                    return dispatch_static_impl<mid + 1, max>(value, std::forward<F>(f));
                }
            }
        }
    }

    /**
     * Runtime -> compile-time dispatch.
     *
     * For a runtime @p value in the inclusive range [ @p min , @p max ], calls
     *     f(std::integral_constant<std::size_t, value>{})
     * so that the value becomes available as a compile-time constant inside the
     * callable. The callable is instantiated for every candidate in the range
     * but invoked exactly once, for the matching value.
     *
     * Throws std::out_of_range if @p value is outside [ @p min , @p max ].
     *
     * The return value of @p f (which must be the same type for every candidate)
     * is forwarded to the caller by value, with guaranteed copy elision; a
     * `void`-returning callable is supported.
     *
     * @tparam min  lowest value handled (inclusive)
     * @tparam max  highest value handled (inclusive)
     * @param  value  the runtime value to dispatch on
     * @param  f      callable taking a std::integral_constant<std::size_t, N>
     */
    template <std::size_t min, std::size_t max, class F>
    constexpr decltype(auto) dispatch_static(std::size_t value, F&& f)
    {
        static_assert(min <= max, "dispatch_static requires min <= max");

        using result_t = decltype(std::forward<F>(f)(std::integral_constant<std::size_t, min>{}));
        static_assert(!std::is_reference_v<result_t>, "dispatch_static: the callable must return by value (or void)");

        return detail::dispatch_static_impl<min, max>(value, std::forward<F>(f));
    }
}
