// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>

#include <xtl/xiterator_base.hpp>

#include "utils.hpp"

// namespace samurai::experimental
namespace samurai
{

    inline auto default_function()
    {
        return [](int, int i)
        {
            return i;
        };
    }

    struct start_end_function
    {
        start_end_function() = default;

        start_end_function(int level, int min_level, int max_level)
            : m_level(level)
            , m_min_shift(min_level - max_level)
            , m_max_shift(max_level - min_level)
        {
        }

        template <class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](int, int i) -> decltype(auto)
            {
                i = start_shift(start_shift(i, m_min_shift), m_max_shift);
                return f(m_level, i);
            };
            return new_f;
        }

        template <class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](int, int i) -> decltype(auto)
            {
                i = end_shift(end_shift(i, m_min_shift), m_max_shift);
                return f(m_level, i);
            };
            return new_f;
        }

        int m_level;
        int m_min_shift;
        int m_max_shift;
    };

    struct start_end_translate_function
    {
        start_end_translate_function() = default;

        start_end_translate_function(int level, int min_level, int max_level, int t)
            : m_level(level)
            , m_min_level(min_level)
            , m_max_level(max_level)
            , m_t(t)
        {
        }

        template <class Func>
        inline auto start(const Func& f) const
        {
            auto new_f = [&, f](int level, int i) -> decltype(auto)
            {
                i = start_shift(start_shift(start_shift(i, level - m_max_level) + m_t, m_min_level - level), m_max_level - m_min_level);
                return f(m_level, i);
            };
            return new_f;
        }

        template <class Func>
        inline auto end(const Func& f) const
        {
            auto new_f = [&, f](int level, int i) -> decltype(auto)
            {
                i = end_shift(end_shift(end_shift(i, level - m_max_level) + m_t, m_min_level - level), m_max_level - m_min_level);
                return f(m_level, i);
            };
            return new_f;
        }

        int m_level;
        int m_min_level;
        int m_max_level;
        int m_t;
    };

    namespace detail

    {
        struct IntervalInfo
        {
            inline auto get_func(auto level, auto min_level, auto max_level, auto)
            {
                return start_end_function(level, min_level, max_level);
            }
        };

    } // namespace detail

    template <class container_>
    class interval_iterator
    {
      public:

        using container_t = container_;
        using value_t     = typename container_t::value_type;
        using iterator_t  = typename container_t::const_iterator;

        interval_iterator(const container_t& data, std::ptrdiff_t start, std::ptrdiff_t end)
            : m_data(data)
            , m_start(start)
            , m_end(end)
        {
        }

        auto begin()
        {
            return m_data.cbegin() + m_start;
        }

        auto end()
        {
            return m_data.cbegin() + m_end;
        }

      private:

        const container_t& m_data;
        std::ptrdiff_t m_start;
        std::ptrdiff_t m_end;
        container_t m_work;
    };

    template <class container_>
    class IntervalVector
    {
      public:

        using container_t = container_;
        using base_t      = interval_iterator<container_t>;
        using iterator_t  = typename base_t::iterator_t;
        using interval_t  = typename base_t::value_t;
        using value_t     = typename interval_t::value_t;
        using function_t  = std::function<int(int, int)>;

        IntervalVector(auto lca_level,
                       auto level,
                       auto max_level,
                       interval_iterator<container_t>&& intervals,
                       function_t&& start_fct,
                       function_t&& end_fct)
            : m_lca_level(static_cast<int>(lca_level))
            , m_shift2dest(max_level - level)
            , m_shift2ref(max_level - static_cast<int>(lca_level))
            , m_intervals(std::move(intervals))
            , m_first(m_intervals.begin())
            , m_last(m_intervals.end())
            , m_current(std::numeric_limits<value_t>::min())
            , m_is_start(true)
            , m_start_fct(std::move(start_fct))
            , m_end_fct(std::move(end_fct))
        {
        }

        IntervalVector(interval_iterator<container_t>&& intervals)
            : m_intervals(std::move(intervals))
            , m_current(sentinel<value_t>)
        {
        }

        auto start(const auto it) const
        {
            auto i = it->start << m_shift2ref;
            return m_start_fct(m_lca_level, i);
        }

        auto end(const auto it) const
        {
            auto i = it->end << m_shift2ref;
            return m_end_fct(m_lca_level, i);
        }

        bool is_in(auto scan) const
        {
            return m_current != sentinel<value_t> && !((scan < m_current) ^ !m_is_start);
        }

        bool is_empty() const
        {
            return m_current == sentinel<value_t>;
        }

        auto min() const
        {
            return m_current;
        }

        auto shift() const
        {
            return m_shift2dest;
        }

        void next(auto scan)
        {
            // std::cout << std::endl;
            // std::cout << "m_current in next: " << m_current << " " << std::numeric_limits<value_t>::min() << std::endl;
            if (m_current == std::numeric_limits<value_t>::min())
            {
                m_current = start(m_first);
                return;
            }

            if (m_current == scan)
            {
                if (m_is_start)
                {
                    m_current = end(m_first);
                    // std::cout << "change m_current: " << m_current << std::endl;
                    while (m_first + 1 != m_last && m_current >= start(m_first + 1))
                    {
                        m_first++;
                        m_current = end(m_first);
                        // std::cout << "update end in while loop: " << m_current << std::endl;
                        // std::cout << "next start: " << start(iop.start_op(m_lca_level, m_first + 1)) << std::boolalpha << " "
                        //           << (m_first + 1 != m_last) << std::endl;
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
                    m_current = start(m_first);
                    // std::cout << "update start: " << m_current << std::endl;
                }
                m_is_start = !m_is_start;
            }
        }

      private:

        int m_lca_level;
        int m_shift2dest;
        int m_shift2ref;
        interval_iterator<container_t> m_intervals;
        iterator_t m_first;
        iterator_t m_last;
        value_t m_current;
        bool m_is_start;
        function_t m_start_fct;
        function_t m_end_fct;
    };

    template <class const_iterator_t>
    class offset_iterator
    {
      public:

        static constexpr std::size_t max_size = 1;
        using iterator_category               = std::forward_iterator_tag;
        using const_iterator                  = const_iterator_t;
        using value_type                      = typename const_iterator_t::value_type;
        using const_reference                 = typename const_iterator_t::reference;
        using pointer                         = typename const_iterator_t::pointer;
        using const_pointer                   = const pointer;

        offset_iterator()
            : p_first({})
            , p_last({})
            , m_current({0, 0})
            , m_size(0)
        {
        }

        offset_iterator(const std::vector<const_iterator_t>& interval_it_begin, const std::vector<const_iterator_t>& interval_it_end)
            : p_first(interval_it_begin)
            , p_last(interval_it_end)
            , m_current({0, 0})
        {
            if (p_first.size() != 1)
            {
                next();
                // std::cout << "first m_current in iterator: " << m_current << std::endl;
            }
            else
            {
                m_current = (interval_it_begin == interval_it_end) ? value_type({0, 0}) : *interval_it_begin[0];
            }
        }

        offset_iterator(const_iterator_t interval_it_begin, const_iterator_t interval_it_end)
            : p_first({interval_it_begin})
            , p_last({interval_it_end})
            , m_current((interval_it_begin == interval_it_end) ? value_type({0, 0}) : *interval_it_begin)
            , m_size(1)
        {
        }

        void next()
        {
            if (m_size == 0)
            {
                return;
            }
            if (m_size == 1)
            {
                if (p_first[0] != p_last[0])
                {
                    p_first[0]++;
                }
                m_current = (p_first[0] != p_last[0]) ? *p_first[0] : value_type({0, 0});
                return;
            }

            using value_t = typename value_type::value_t;

            if (p_first != p_last)
            {
                auto start = std::numeric_limits<value_t>::max();
                auto end   = std::numeric_limits<value_t>::min();

                for (std::size_t i = 0; i < p_first.size(); ++i)
                {
                    if (p_first[i] != p_last[i])
                    {
                        if (start >= p_first[i]->start)
                        {
                            start = p_first[i]->start;
                            end   = std::max(end, p_first[i]->end);
                            p_first[i]++;
                        }
                    }
                }

                bool unchanged = false;
                while (!unchanged)
                {
                    unchanged = true;
                    for (std::size_t i = 0; i < p_first.size(); ++i)
                    {
                        if (p_first[i] != p_last[i] && p_first[i]->start <= end)
                        {
                            end = std::max(end, p_first[i]->end);
                            p_first[i]++;
                            unchanged = false;
                        }
                    }
                }

                m_current = {start, end};
            }
            else
            {
                m_current = {0, 0};
            }
        }

        offset_iterator& operator++()
        {
            next();
            return *this;
        }

        offset_iterator operator++(int)
        {
            offset_iterator temp = *this;
            ++(*this);
            return temp;
        }

        const_reference operator*() const
        {
            return m_current;
        }

        const_pointer operator->() const
        {
            return &m_current;
        }

        bool operator==(const offset_iterator& other) const
        {
            return p_first == other.p_first && m_current == other.m_current;
        }

        bool operator!=(const offset_iterator& other) const
        {
            return !(*this == other);
        }

      private:

        // std::vector<const_iterator_t> p_first;
        // std::vector<const_iterator_t> p_last;
        std::array<const_iterator_t, max_size> p_first;
        std::array<const_iterator_t, max_size> p_last;
        value_type m_current;
        std::size_t m_size;
    };

    template <class const_iterator_t>
    auto operator+(const offset_iterator<const_iterator_t>& it, int i)
    {
        offset_iterator<const_iterator_t> temp = it;
        for (int ii = 0; ii < i; ++ii)
        {
            temp++;
        }
        return temp;
    }
}