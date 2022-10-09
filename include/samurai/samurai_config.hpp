// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#define SAMURAI_VERSION_MAJOR 0
#define SAMURAI_VERSION_MINOR 1
#define SAMURAI_VERSION_PATCH 0

namespace samurai
{
    template<class TValue, class TIndex>
    struct Interval;

    namespace default_config
    {
        static constexpr std::size_t max_level = 20;
        static constexpr std::size_t ghost_width = 1;
        static constexpr std::size_t graduation_width = 1;
        static constexpr std::size_t prediction_order = 1;

        using index_t = signed long long int;
        using value_t = int;
        using interval_t = Interval<value_t, index_t>;
    }
}