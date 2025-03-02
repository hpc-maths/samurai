// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "utils.hpp"

namespace samurai
{

    inline auto default_function()
    {
        return [](auto, auto i)
        {
            return i;
        };
    }

    template <std::size_t dim>
    struct start_end_function
    {
        auto& operator()(std::size_t level, std::size_t min_level, std::size_t max_level)
        {
            m_level     = level;
            m_min_shift = static_cast<int>(min_level) - static_cast<int>(max_level);
            m_max_shift = static_cast<int>(max_level) - static_cast<int>(min_level);
            return *this;
        }

        template <std::size_t, class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](auto, auto i)
            {
                i = start_shift(start_shift(i, m_min_shift), m_max_shift);
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t, class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](auto, auto i)
            {
                i = end_shift(end_shift(i, m_min_shift), m_max_shift);
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t, class Func>
        inline auto goback(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                // std::cout << "go_back previous i: " << i << std::endl;
                // std::cout << "previous level: " << level << " current level: " << m_level << std::endl;
                i = start_shift(f(m_level, i), static_cast<int>(level) - static_cast<int>(m_level));
                // std::cout << " next i " << i << std::endl << std::endl;
                return i;
            };
            return new_f;
        }

        std::size_t m_level;
        int m_min_shift;
        int m_max_shift;
    };

    template <std::size_t dim>
    struct start_end_translate_function
    {
        using container_t = xt::xtensor_fixed<int, xt::xshape<dim>>;

        start_end_translate_function(const container_t& t)
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

        template <std::size_t d, class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                i = start_shift(start_shift(start_shift(i, static_cast<int>(level) - static_cast<int>(m_max_level)) + m_t[d - 1],
                                            static_cast<int>(m_min_level) - static_cast<int>(level)),
                                static_cast<int>(m_max_level) - static_cast<int>(m_min_level));
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t d, class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                i = end_shift(end_shift(end_shift(i, static_cast<int>(level) - static_cast<int>(m_max_level)) + m_t[d - 1],
                                        static_cast<int>(m_min_level) - static_cast<int>(level)),
                              static_cast<int>(m_max_level) - static_cast<int>(m_min_level));
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t d, class Func>
        inline auto goback(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                // std::cout << "go_back translate previous i: " << i << " translation: " << m_t[d - 1] << " "
                //           << start_shift(m_t[d - 1], m_level - level) << std::endl;
                // std::cout << "previous level: " << level << " current level: " << m_level << std::endl;
                // i = start_shift(f(m_level, i - start_shift(m_t[d - 1], m_level - level)), level - m_level);
                i = start_shift(f(m_level, i), static_cast<int>(level) - static_cast<int>(m_level)) - m_t[d - 1];
                // std::cout << " translate next i " << i << std::endl << std::endl;
                return i;
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
        start_end_contraction_function(int c)
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

        template <std::size_t d, class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                i = start_shift(start_shift(start_shift(i, static_cast<int>(level) - static_cast<int>(m_max_level)) - m_c,
                                            static_cast<int>(m_min_level) - static_cast<int>(level)),
                                static_cast<int>(m_max_level) - static_cast<int>(m_min_level));
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t d, class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                i = end_shift(end_shift(end_shift(i, static_cast<int>(level) - static_cast<int>(m_max_level)) - m_c,
                                        static_cast<int>(m_min_level) - static_cast<int>(level)),
                              static_cast<int>(m_max_level) - static_cast<int>(m_min_level));
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t d, class Func>
        inline auto goback(const Func& f) const
        {
            auto new_f = [&, f](auto level, auto i)
            {
                // std::cout << "go_back translate previous i: " << i << " translation: " << m_t[d - 1] << " "
                //           << start_shift(m_t[d - 1], m_level - level) << std::endl;
                // std::cout << "previous level: " << level << " current level: " << m_level << std::endl;
                // i = start_shift(f(m_level, i - start_shift(m_t[d - 1], m_level - level)), level - m_level);
                i = start_shift(f(m_level, i), static_cast<int>(level) - static_cast<int>(m_level)) + m_c;
                // std::cout << " translate next i " << i << std::endl << std::endl;
                return i;
            };
            return new_f;
        }

        std::size_t m_level;
        std::size_t m_min_level;
        std::size_t m_max_level;
        int m_c;
    };
}
