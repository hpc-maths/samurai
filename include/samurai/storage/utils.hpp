// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../samurai_config.hpp"

namespace samurai
{
    namespace detail
    {
        template <std::size_t size, bool SOA, layout_type L>
        struct static_size_first : std::false_type
        {
        };

        template <std::size_t size>
            requires(size > 1)
        struct static_size_first<size, true, layout_type::row_major> : std::true_type
        {
        };

        template <std::size_t size>
            requires(size > 1)
        struct static_size_first<size, false, layout_type::column_major> : std::true_type
        {
        };

        template <std::size_t size, bool SOA, layout_type L>
        static constexpr bool static_size_first_v = static_size_first<size, SOA, L>::value;
    }

    template <class T>
    struct range_t
    {
        T start;
        T end;
        T step = 1;
    };
}
