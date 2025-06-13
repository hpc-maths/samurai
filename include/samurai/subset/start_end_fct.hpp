// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "utils.hpp"

namespace samurai
{

    inline auto default_function()
    {
        return [](auto, auto i, auto)
        {
            return i;
        };
    }

    inline auto default_function_()
    {
        return [](auto level, auto i)
        {
            return std::make_pair(level, i);
        };
    }

    struct start_end_local_function
    {
        start_end_local_function(std::size_t min_level, std::size_t max_level)
            : m_max2min(static_cast<int>(max_level) - static_cast<int>(min_level))
        {
        }

        template <class interval_t>
        inline auto start(const interval_t& i) const
        {
            using value_t = typename interval_t::value_t;
            // std::cout << "start: i.start = " << i.start << ", m_max2min = " << m_max2min << std::endl;
            return static_cast<value_t>((i.start >> m_max2min) << m_max2min);
        }

        template <class interval_t>
        inline auto end(const interval_t& i) const
        {
            using value_t = typename interval_t::value_t;
            // std::cout << "end: i.end = " << i.end << ", m_max2min = " << m_max2min << std::endl;
            return static_cast<value_t>((((i.end - 1) >> m_max2min) + 1) << m_max2min);
        }

        template <class interval_t>
        inline void operator()(interval_t& i) const
        {
            i.start = start(i);
            i.end   = end(i);
        }

        int m_max2min;
    };

    template <std::size_t dim>
    struct start_end_function
    {
        template <std::size_t d>
        auto get_local_function(std::size_t /*level*/, std::size_t min_level, std::size_t max_level)
        {
            return start_end_local_function(min_level, max_level);
        }

        template <std::size_t, bool end = false, class Func>
        inline auto goback(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                auto [prev_lev, v] = f(level, i);

                int min_shift = static_cast<int>(m_min_level) - static_cast<int>(prev_lev);
                int max_shift = static_cast<int>(m_level) - static_cast<int>(m_min_level);

                if constexpr (end)
                {
                    i = end_shift(end_shift(v, min_shift), max_shift);
                }
                else
                {
                    // std::cout << "goback: level = " << level << ", i = " << i << ", prev_lev = " << prev_lev << ", v = " << v <<
                    // std::endl;
                    i = start_shift(start_shift(v, min_shift), max_shift);
                    // std::cout << "goback: after start_shift, i = " << i << std::endl;
                }

                return std::make_pair(m_level, i);
            };
            return new_f;
        }

        std::size_t m_level;
        int m_shift;
        std::size_t m_min_level;
    };

    struct start_end_translate_local_function
    {
        start_end_translate_local_function(std::size_t level, std::size_t min_level, std::size_t max_level, int translation)
            : m_max2curr(static_cast<int>(max_level) - static_cast<int>(level))
            , m_curr2min(static_cast<int>(level) - static_cast<int>(min_level))
            , m_min2max(static_cast<int>(max_level) - static_cast<int>(min_level))
            , m_translation(translation)
        {
        }

        template <class interval_t>
        inline auto start(const interval_t& i) const
        {
            using value_t = typename interval_t::value_t;
            // std::cout << "start: i.start = " << i.start << ", m_max2curr = " << m_max2curr << ", m_translation = " << m_translation
            //           << ", m_curr2min = " << m_curr2min << ", m_min2max = " << m_min2max
            //           << " value: " << static_cast<value_t>((((i.start >> m_max2curr) + m_translation) >> m_curr2min) << m_min2max)
            //           << std::endl;
            return static_cast<value_t>((((i.start >> m_max2curr) + m_translation) >> m_curr2min) << m_min2max);
        }

        template <class interval_t>
        inline auto end(const interval_t& i) const
        {
            using value_t = typename interval_t::value_t;
            // std::cout << "end: i.end = " << i.end << ", m_max2curr = " << m_max2curr << ", m_translation = " << m_translation
            //           << ", m_curr2min = " << m_curr2min << ", m_min2max = " << m_min2max
            //           << " value: " << static_cast<value_t>((((((i.end - 1) >> m_max2curr) + m_translation) >> m_curr2min) + 1) <<
            //           m_min2max)
            //           << std::endl;
            return static_cast<value_t>((((((i.end - 1) >> m_max2curr) + m_translation) >> m_curr2min) + 1) << m_min2max);
        }

        template <class interval_t>
        inline void operator()(interval_t& i) const
        {
            i.start = start(i);
            i.end   = end(i);
        }

        int m_max2curr;
        int m_curr2min;
        int m_min2max;
        int m_translation;
    };

    template <std::size_t dim>
    struct start_end_translate_function
    {
        using container_t = xt::xtensor_fixed<int, xt::xshape<dim>>;

        explicit start_end_translate_function(const container_t& t)
            : m_level(0)
            , m_min_level(0)
            , m_max_level(0)
            , m_t(t)
        {
        }

        template <std::size_t d>
        auto get_local_function(auto level, auto min_level, auto max_level)
        {
            return start_end_translate_local_function(level, min_level, max_level, m_t[d - 1]);
        }

        template <std::size_t d, bool end = false, class Func>
        inline auto goback(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                auto [prev_lev, v] = f(level, i);

                auto min_shift = static_cast<int>(m_min_level) - static_cast<int>(prev_lev);
                auto max_shift = static_cast<int>(m_level) - static_cast<int>(m_min_level);

                if constexpr (end)
                {
                    i = end_shift(end_shift(v, min_shift), max_shift) - m_t[d - 1];
                }
                else
                {
                    // std::cout << "translate goback: level = " << level << ", i = " << i << ", prev_lev = " << prev_lev << ", v = " << v
                    //           << " m_t = " << m_t[d - 1] << std::endl;
                    i = start_shift(start_shift(v, min_shift), max_shift) - m_t[d - 1];
                    // std::cout << "translate goback: after start_shift, i = " << i << std::endl;
                }

                return std::make_pair(m_level, i);
            };
            return new_f;
        }

        std::size_t m_level;
        std::size_t m_min_level;
        std::size_t m_max_level;
        xt::xtensor_fixed<int, xt::xshape<dim>> m_t;
    };
}
