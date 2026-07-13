// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace samurai
{
    namespace detail
    {
        template <std::size_t min, class F, std::size_t... Is>
        constexpr decltype(auto) dispatch_static_impl(std::size_t value, F&& f, std::index_sequence<Is...>)
        {
            using result_t = decltype(std::forward<F>(f)(std::integral_constant<std::size_t, min>{}));

            if constexpr (std::is_void_v<result_t>)
            {
                // Fold over the candidates: the ternary short-circuits so `f` is
                // invoked at most once (for the branch where `value == min + Is`).
                const bool matched = ((value == min + Is ? (std::forward<F>(f)(std::integral_constant<std::size_t, min + Is>{}), true) : false)
                                      || ...);
                if (!matched)
                {
                    throw std::out_of_range("dispatch_static: value out of range");
                }
            }
            else
            {
                static_assert(!std::is_reference_v<result_t>, "dispatch_static: the callable must return by value (or void)");

                std::optional<result_t> result;
                const bool matched = ((value == min + Is
                                           ? (result.emplace(std::forward<F>(f)(std::integral_constant<std::size_t, min + Is>{})), true)
                                           : false)
                                      || ...);
                if (!matched)
                {
                    throw std::out_of_range("dispatch_static: value out of range");
                }
                return static_cast<result_t>(std::move(*result));
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
     * is forwarded to the caller; a `void`-returning callable is supported.
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
        return detail::dispatch_static_impl<min>(value, std::forward<F>(f), std::make_index_sequence<max - min + 1>{});
    }
}
