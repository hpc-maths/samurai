// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "layout_config.hpp"

namespace samurai
{
    namespace detail
    {
        template <std::size_t size, bool SOA, bool can_collapse, layout_type L>
        struct static_size_first : std::false_type
        {
        };

        template <std::size_t size, bool can_collapse>
            requires(size > 1)
        struct static_size_first<size, true, can_collapse, layout_type::row_major> : std::true_type
        {
        };

        template <std::size_t size, bool can_collapse>
            requires(size > 1)
        struct static_size_first<size, false, can_collapse, layout_type::column_major> : std::true_type
        {
        };

        template <bool SOA, layout_type L>
        struct static_size_first<1, SOA, false, L> : std::true_type
        {
        };

        template <std::size_t size, bool SOA, bool can_collapse, layout_type L>
        static constexpr bool static_size_first_v = static_size_first<size, SOA, can_collapse, L>::value;
    }

    template <class T>
    struct range_t
    {
        T start;
        T end;
        T step = 1;
    };
}
