// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    enum class layout_type
    {
        /*! row major layout_type */
        row_major = 0x00,
        /*! column major layout_type */
        column_major = 0x01
    };

#ifndef SAMURAI_DEFAULT_LAYOUT
#define SAMURAI_DEFAULT_LAYOUT ::samurai::layout_type::row_major
// #define SAMURAI_DEFAULT_LAYOUT ::samurai::layout_type::column_major
#endif

    static constexpr bool disable_color = true;

    template <class TValue, class TIndex>
    struct Interval;

    namespace default_config
    {
        static constexpr std::size_t max_level        = 20;
        static constexpr std::size_t ghost_width      = 1;
        static constexpr std::size_t graduation_width = 1;
        static constexpr std::size_t prediction_order = 1;

        using index_t    = signed long long int;
        using value_t    = int;
        using interval_t = Interval<value_t, index_t>;
    }
}
