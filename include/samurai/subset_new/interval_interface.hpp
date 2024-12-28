// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <ios>
#include <limits>

#include <xtl/xiterator_base.hpp>

#include "utils.hpp"

namespace samurai::experimental
{
    namespace detail
    {
        struct IntervalInfo
        {
            inline auto start_op(auto, const auto it)
            {
                return it->start;
            }

            inline auto end_op(auto, const auto it)
            {
                return it->end;
            }

            void set_level(auto)
            {
            }
        };

    } // namespace detail

    template <class iterator>
    class IntervalVector
    {
      public:

        using iterator_t = iterator;
        using interval_t = typename iterator_t::value_type;
        using value_t    = typename interval_t::value_t;

        IntervalVector(auto lca_level, auto level, auto min_level, auto max_level, iterator_t begin, iterator_t end)
            : m_min_shift(min_level - static_cast<int>(lca_level))
            , m_max_shift(max_level - min_level)
            , m_shift(max_level - level)
            , m_lca_level(static_cast<int>(lca_level))
            , m_first(begin)
            , m_last(end)
            , m_current(std::numeric_limits<value_t>::min())
            , m_is_start(true)
        {
        }

        IntervalVector()
            : m_current(sentinel<value_t>)
        {
        }

        auto start(const auto i) const
        {
            return start_shift(start_shift(i, m_min_shift), m_max_shift);
        }

        auto end(const auto i) const
        {
            return end_shift(end_shift(i, m_min_shift), m_max_shift);
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
            return m_shift;
        }

        template <class IntervalOp = detail::IntervalInfo>
        void next(auto scan, IntervalOp iop = {})
        {
            // std::cout << "m_current in next: " << m_current << " " << std::numeric_limits<value_t>::min() << std::endl;
            if (m_current == std::numeric_limits<value_t>::min())
            {
                m_current = start(iop.start_op(m_lca_level, m_first));
                // std::cout << "first m_current: " << m_current << std::endl;
                return;
            }

            if (m_current == scan)
            {
                if (m_is_start)
                {
                    m_current = end(iop.end_op(m_lca_level, m_first));
                    // std::cout << "change m_current: " << m_current << std::endl;
                    while (m_first + 1 != m_last && m_current >= start(iop.start_op(m_lca_level, m_first + 1)))
                    {
                        m_first++;
                        m_current = end(iop.end_op(m_lca_level, m_first));
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
                    m_current = start(iop.start_op(m_lca_level, m_first));
                    // std::cout << "update start: " << m_current << std::endl;
                }
                m_is_start = !m_is_start;
            }
        }

      private:

        int m_min_shift;
        int m_max_shift;
        int m_shift;
        int m_lca_level;
        iterator_t m_first;
        iterator_t m_last;
        value_t m_current;
        bool m_is_start;
    };

    template <class const_iterator_t>
    class offset_iterator
    {
      public:

        using iterator_category = std::forward_iterator_tag;
        using const_iterator    = const_iterator_t;
        using value_type        = typename const_iterator_t::value_type;
        using const_reference   = typename const_iterator_t::reference;
        using pointer           = typename const_iterator_t::pointer;
        using const_pointer     = const pointer;

        offset_iterator() = default;

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
        }

        offset_iterator(const_iterator_t interval_it_begin, const_iterator_t interval_it_end)
            : p_first({interval_it_begin})
            , p_last({interval_it_end})
            , m_current(*interval_it_begin)
        {
        }

        void next()
        {
            if (p_first.size() == 1)
            {
                if (p_first[0] != p_last[0])
                {
                    p_first[0]++;
                }
                m_current = *p_first[0];
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

        std::vector<const_iterator_t> p_first;
        std::vector<const_iterator_t> p_last;
        value_type m_current;
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