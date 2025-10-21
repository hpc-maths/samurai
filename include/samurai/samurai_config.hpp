// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * @brief Conditional inlining macros controlled by CMake.
 *
 * When SAMURAI_ENABLE_INLINE is defined (default ON):
 * - SAMURAI_INLINE expands to the standard `inline` keyword.
 * - SAMURAI_FORCE_INLINE expands to compiler-specific forced inlining hints.
 *
 * When SAMURAI_ENABLE_INLINE is NOT defined:
 * - Both macros expand to nothing, disabling inlining for analysis/debugging.
 *
 * Compiler support for SAMURAI_FORCE_INLINE:
 * - GCC/Clang: `inline __attribute__((always_inline))`
 * - MSVC: `__forceinline`
 * - Others: falls back to `inline`
 */
#if defined(SAMURAI_FORCE_INLINE)
// Precedence: if the force-inline flag is set, use forced inlining
#if defined(_MSC_VER)
#define SAMURAI_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SAMURAI_INLINE inline __attribute__((always_inline))
#else
#define SAMURAI_INLINE inline
#endif
#elif defined(SAMURAI_ENABLE_INLINE)
// Enable standard inlining only
#define SAMURAI_INLINE inline
#else
// Disable all inlining
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

        static constexpr bool projection_with_list_of_intervals = true;
        static constexpr bool use_native_expand                 = false;

        using index_t    = signed long long int;
        using value_t    = int;
        using interval_t = Interval<value_t, index_t>;

        inline auto default_prediction_fn = [](auto& new_field, const auto& old_field) // cppcheck-suppress constParameterReference
        {
            constexpr std::size_t pred_order = std::decay_t<decltype(new_field)>::mesh_t::config::prediction_stencil_radius;
            return prediction<pred_order, true>(new_field, old_field);
        };
    }
}
