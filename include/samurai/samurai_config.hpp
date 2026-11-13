// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <type_traits>

/**
 * @brief Inlining macros.
 *
 * SAMURAI_INLINE follows the CMake options:
 * - SAMURAI_FORCE_INLINE=ON: forced inlining, as SAMURAI_ALWAYS_INLINE.
 * - SAMURAI_ENABLE_INLINE=ON (default): the standard `inline` keyword.
 * - both OFF: nothing, for analysis and debugging.
 *
 * SAMURAI_ALWAYS_INLINE forces inlining whatever the CMake options. Use it on
 * the small functions a flux function calls for every face:
 * - GCC/Clang: `inline __attribute__((always_inline))`
 * - MSVC: `__forceinline`
 * - others: falls back to `inline`
 */
#if defined(_MSC_VER)
#define SAMURAI_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SAMURAI_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SAMURAI_ALWAYS_INLINE inline
#endif

#if defined(SAMURAI_FORCE_INLINE)
#define SAMURAI_INLINE SAMURAI_ALWAYS_INLINE
#elif defined(SAMURAI_ENABLE_INLINE)
#define SAMURAI_INLINE inline
#else
#define SAMURAI_INLINE
#endif

namespace samurai
{
    static constexpr bool disable_color = true;

    template <class TValue, class TIndex>
    struct Interval;

    template <std::size_t order, bool dest_on_level, class T1, class T2>
    SAMURAI_INLINE auto prediction(T1& field_dest, const T2& field_src);

    namespace default_config
    {
        static constexpr std::size_t max_level         = 20;
        static constexpr int ghost_width               = 1;
        static constexpr int graduation_width          = 1;
        static constexpr int prediction_stencil_radius = 1;

        static constexpr bool use_native_expand = true;

        using index_t    = signed long long int;
        using value_t    = int;
        using interval_t = Interval<value_t, index_t>;

        inline auto default_prediction_fn = [](auto& new_field, const auto& old_field) // cppcheck-suppress constParameterReference
        {
            constexpr std::size_t pred_order = std::decay_t<decltype(new_field)>::mesh_t::config_t::prediction_stencil_radius;
            return prediction<pred_order, true>(new_field, old_field);
        };
    }
}
