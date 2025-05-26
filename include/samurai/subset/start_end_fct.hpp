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

    template <std::size_t dim>
    struct start_end_function
    {
        auto& operator()(std::size_t level, std::size_t min_level, std::size_t max_level)
        {
            m_level     = level;
            m_min_level = min_level;
            m_shift     = static_cast<int>(max_level) - static_cast<int>(min_level);
            return *this;
        }

        template <std::size_t, bool from_diff_op = false, class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](auto, auto i, auto dec)
            {
                if constexpr (from_diff_op)
                {
                    dec = 1;
                }
                int value = (((i - dec) >> m_shift) << m_shift) + dec;
                return f(m_level, value, dec);
            };
            return new_f;
        }

        template <std::size_t, bool from_diff_op = false, class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](auto, auto i, auto dec)
            {
                if constexpr (from_diff_op)
                {
                    dec = 0;
                }
                int value = (((i - dec) >> m_shift) + dec) << m_shift;
                return f(m_level, value, dec);
            };
            return new_f;
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
                    i = start_shift(start_shift(v, min_shift), max_shift);
                }

                return std::make_pair(m_level, i);
            };
            return new_f;
        }

        std::size_t m_level;
        int m_shift;
        std::size_t m_min_level;
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

        auto& operator()(auto level, auto min_level, auto max_level)
        {
            m_level     = level;
            m_min_level = min_level;
            m_max_level = max_level;
            return *this;
        }

        template <std::size_t d, bool from_diff_op = false, class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i, auto dec)
            {
                int max2curr = static_cast<int>(m_max_level) - static_cast<int>(level);
                int curr2min = static_cast<int>(level) - static_cast<int>(m_min_level);
                int min2max  = static_cast<int>(m_max_level) - static_cast<int>(m_min_level);

                if constexpr (from_diff_op)
                {
                    dec = 1;
                }
                int value = (((((i - dec) >> max2curr) + m_t[d - 1]) >> curr2min) + dec) << min2max;

                return f(m_level, value, dec);
            };
            return new_f;
        }

        template <std::size_t d, bool from_diff_op = false, class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i, auto dec)
            {
                int max2curr = static_cast<int>(m_max_level) - static_cast<int>(level);
                int curr2min = static_cast<int>(level) - static_cast<int>(m_min_level);
                int min2max  = static_cast<int>(m_max_level) - static_cast<int>(m_min_level);

                if constexpr (from_diff_op)
                {
                    dec = 0;
                }
                int value = (((((i - dec) >> max2curr) + m_t[d - 1]) >> curr2min) + dec) << min2max;

                return f(m_level, value, dec);
            };
            return new_f;
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
                    i = start_shift(start_shift(v, min_shift), max_shift) - m_t[d - 1];
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

    template <std::size_t dim>
    struct start_end_contraction_function
    {
        explicit start_end_contraction_function(int c)
            : m_level(0)
            , m_min_level(0)
            , m_max_level(0)
            , m_c(c)
        {
        }

        auto& operator()(auto level, auto min_level, auto max_level)
        {
            m_level     = level;
            m_min_level = min_level;
            m_max_level = max_level;
            return *this;
        }

        template <std::size_t d, bool from_diff_op = false, class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i, auto dec)
            {
                int max2curr = static_cast<int>(m_max_level) - static_cast<int>(level);
                int curr2min = static_cast<int>(level) - static_cast<int>(m_min_level);
                int min2max  = static_cast<int>(m_max_level) - static_cast<int>(m_min_level);

                if constexpr (from_diff_op)
                {
                    dec = 1;
                }
                int value = (((((i - dec) >> max2curr) - m_c) >> curr2min) + dec) << min2max;
                return f(m_level, value, dec);
            };
            return new_f;
        }

        template <std::size_t d, bool from_diff_op = false, class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i, auto dec)
            {
                int max2curr = static_cast<int>(m_max_level) - static_cast<int>(level);
                int curr2min = static_cast<int>(level) - static_cast<int>(m_min_level);
                int min2max  = static_cast<int>(m_max_level) - static_cast<int>(m_min_level);

                if constexpr (from_diff_op)
                {
                    dec = 0;
                }
                int value = (((((i - dec) >> max2curr) - m_c) >> curr2min) + dec) << min2max;
                return f(m_level, value, dec);
            };
            return new_f;
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
                    i = end_shift(end_shift(v, min_shift), max_shift) + m_c;
                }
                else
                {
                    i = start_shift(start_shift(v, min_shift), max_shift) + m_c;
                }

                return std::make_pair(m_level, i);
            };
            return new_f;
        }

        std::size_t m_level;
        std::size_t m_min_level;
        std::size_t m_max_level;
        int m_c;
    };
}
