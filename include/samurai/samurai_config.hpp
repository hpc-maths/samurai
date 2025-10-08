// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    static constexpr bool disable_color = true;

    template <class TValue, class TIndex>
    struct Interval;

    template <std::size_t order, bool dest_on_level, class T1, class T2>
    inline auto prediction(T1& field_dest, const T2& field_src);

    namespace default_config
    {
        static constexpr std::size_t max_level        = 20;
        static constexpr std::size_t ghost_width      = 1;
        static constexpr std::size_t graduation_width = 1;
        static constexpr std::size_t prediction_order = 1;

        static constexpr bool prediction_with_list_of_intervals = false;

        using index_t    = signed long long int;
        using value_t    = int;
        using interval_t = Interval<value_t, index_t>;

        inline auto default_prediction_fn = [](auto& new_field, const auto& old_field) // cppcheck-suppress constParameterReference
        {
            constexpr std::size_t pred_order = std::decay_t<decltype(new_field)>::mesh_t::config::prediction_order;
            return prediction<pred_order, true>(new_field, old_field);
        };
    }
}
