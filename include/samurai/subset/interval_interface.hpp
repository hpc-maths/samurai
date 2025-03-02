// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

#include <xtensor/xfixed.hpp>

#include "utils.hpp"

// namespace samurai::experimental
namespace samurai
{

    inline auto default_function()
    {
        return [](int, int i)
        {
            // std::cout << "default value " << i << std::endl;
            return i;
        };
    }

    template <std::size_t dim>
    struct start_end_function
    {
        start_end_function() = default;

        auto& operator()(int level, int min_level, int max_level)
        {
            m_level     = level;
            m_min_shift = min_level - max_level;
            m_max_shift = max_level - min_level;
            return *this;
        }

        template <std::size_t, class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](int, int i)
            {
                i = start_shift(start_shift(i, m_min_shift), m_max_shift);
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t, class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](int, int i)
            {
                i = end_shift(end_shift(i, m_min_shift), m_max_shift);
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t, class Func>
        inline auto goback(const Func& f) const
        {
            auto new_f = [&, f](int level, int i)
            {
                // std::cout << "go_back previous i: " << i << std::endl;
                // std::cout << "previous level: " << level << " current level: " << m_level << std::endl;
                i = start_shift(f(m_level, i), level - m_level);
                // std::cout << " next i " << i << std::endl << std::endl;
                return i;
            };
            return new_f;
        }

        int m_level;
        int m_min_shift;
        int m_max_shift;
    };

    template <std::size_t dim>
    struct start_end_translate_function
    {
        using container_t = xt::xtensor_fixed<int, xt::xshape<dim>>;

        start_end_translate_function(const container_t& t)
            : m_t(t)
        {
        }

        auto& operator()(int level, int min_level, int max_level)
        {
            m_level     = level;
            m_min_level = min_level;
            m_max_level = max_level;
            return *this;
        }

        template <std::size_t d, class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](int level, int i)
            {
                i = start_shift(start_shift(start_shift(i, level - m_max_level) + m_t[d - 1], m_min_level - level),
                                m_max_level - m_min_level);
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t d, class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](int level, int i)
            {
                i = end_shift(end_shift(end_shift(i, level - m_max_level) + m_t[d - 1], m_min_level - level), m_max_level - m_min_level);
                return f(m_level, i);
            };
            return new_f;
        }

        template <std::size_t d, class Func>
        inline auto goback(const Func& f) const
        {
            auto new_f = [&, f](int level, int i)
            {
                // std::cout << "go_back translate previous i: " << i << " translation: " << m_t[d - 1] << " "
                //           << start_shift(m_t[d - 1], m_level - level) << std::endl;
                // std::cout << "previous level: " << level << " current level: " << m_level << std::endl;
                // i = start_shift(f(m_level, i - start_shift(m_t[d - 1], m_level - level)), level - m_level);
                i = start_shift(f(m_level, i), level - m_level) - m_t[d - 1];
                // std::cout << " translate next i " << i << std::endl << std::endl;
                return i;
            };
            return new_f;
        }

        int m_level;
        int m_min_level;
        int m_max_level;
        xt::xtensor_fixed<int, xt::xshape<dim>> m_t;
    };

    template <class container_>
    class IntervalIterator
    {
      public:

        using container_t = container_;
        using value_t     = typename container_t::value_type;
        using iterator_t  = typename container_t::const_iterator;

        IntervalIterator(const container_t& data, std::ptrdiff_t start, std::ptrdiff_t end)
            : m_data(data)
            , m_start(start)
            , m_end(end)
            , m_work(data)
            , m_take_data(true)
        {
        }

        IntervalIterator(const container_t& data, const container_t& w)
            : m_data(data)
            , m_start(0)
            , m_end(0)
            , m_work(w)
            , m_take_data(false)
        {
        }

        inline auto begin()
        {
            return (m_take_data) ? m_data.cbegin() + m_start : m_work.cbegin();
        }

        inline auto end()
        {
            return (m_take_data) ? m_data.cbegin() + m_end : m_work.cend();
        }

      private:

        const container_t& m_data;
        std::ptrdiff_t m_start;
        std::ptrdiff_t m_end;
        const container_t& m_work;
        bool m_take_data;
    };

    template <class container_>
    class IntervalVector
    {
      public:

        using container_t = container_;
        using base_t      = IntervalIterator<container_t>;
        using iterator_t  = typename base_t::iterator_t;
        using interval_t  = typename base_t::value_t;
        using value_t     = typename interval_t::value_t;

        IntervalVector(auto lca_level, auto level, auto max_level, IntervalIterator<container_t>&& intervals)
            : m_lca_level(static_cast<int>(lca_level))
            , m_shift2dest(max_level - level)
            , m_shift2ref(max_level - static_cast<int>(lca_level))
            , m_intervals(std::move(intervals))
            , m_first(m_intervals.begin())
            , m_last(m_intervals.end())
            , m_current(std::numeric_limits<value_t>::min())
            , m_is_start(true)
        {
        }

        IntervalVector(IntervalIterator<container_t>&& intervals)
            : m_intervals(std::move(intervals))
            , m_current(sentinel<value_t>)
        {
        }

        template <class Func>
        inline auto start(const auto it, Func& start_fct) const
        {
            auto i = it->start << m_shift2ref;
            return start_fct(m_lca_level, i);
        }

        template <class Func>
        inline auto end(const auto it, Func& end_fct) const
        {
            auto i = it->end << m_shift2ref;
            return end_fct(m_lca_level, i);
        }

        inline bool is_in(auto scan) const
        {
            return m_current != sentinel<value_t> && !((scan < m_current) ^ !m_is_start);
        }

        inline bool is_empty() const
        {
            return m_current == sentinel<value_t>;
        }

        inline auto min() const
        {
            return m_current;
        }

        inline auto shift() const
        {
            return m_shift2dest;
        }

        template <class StartEnd>
        inline void next(auto scan, StartEnd& start_and_stop)
        {
            auto& [start_fct, end_fct] = start_and_stop;
            // std::cout << std::endl;
            // std::cout << "m_current in next: " << m_current << " " << std::numeric_limits<value_t>::min() << std::endl;
            if (m_current == std::numeric_limits<value_t>::min())
            {
                m_current = start(m_first, start_fct);
                // std::cout << "first start " << m_current << std::endl;
                return;
            }

            if (m_current == scan)
            {
                if (m_is_start)
                {
                    m_current = end(m_first, end_fct);
                    // std::cout << "change m_current: " << m_current << std::endl;
                    while (m_first + 1 != m_last && m_current >= start(m_first + 1, start_fct))
                    {
                        m_first++;
                        m_current = end(m_first, end_fct);
                        // std::cout << "update end in while loop: " << m_current << std::endl;
                        // std::cout << "next start: " << start(m_first + 1, start_fct) << std::boolalpha << " " << (m_first + 1 != m_last)
                        //           << std::endl;
                    }
                    // std::cout << "update end: " << m_current << std::endl;
                }
                else
                {
                    m_first++;

                    if (m_first == m_last)
                    {
                        m_current = sentinel<value_t>;
                        return;
                    }
                    m_current = start(m_first, start_fct);
                    // std::cout << "update start: " << m_current << std::endl;
                }
                m_is_start = !m_is_start;
            }
        }

      private:

        int m_lca_level;
        int m_shift2dest;
        int m_shift2ref;
        IntervalIterator<container_t> m_intervals;
        iterator_t m_first;
        iterator_t m_last;
        value_t m_current;
        bool m_is_start;
    };
}