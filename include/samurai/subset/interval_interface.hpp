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
    struct node_impl
    {
        node_impl()                               = default;
        virtual int start(int level, int i) const = 0;
        virtual int end(int level, int i) const   = 0;
        virtual int level() const                 = 0;

        virtual ~node_impl() = default;

        node_impl(const node_impl&)            = delete;
        node_impl& operator=(const node_impl&) = delete;
        node_impl(node_impl&&)                 = delete;
        node_impl& operator=(node_impl&&)      = delete;
    };

    struct self_node : public node_impl
    {
        self_node(int level, int min_level, int max_level)
            : node_impl()
            , m_level(level)
            , m_min_level(min_level)
            , m_max_level(max_level)
        {
        }

        virtual int start(int, int i) const override
        {
            auto min_shift = m_min_level - m_max_level;
            auto max_shift = m_max_level - m_min_level;
            // std::cout << fmt::format(" (level: {}, m_level: {}, min_level: {}, max_level: {}, min_shift: {}, max_shift: {}) ",
            //                          level,
            //                          m_level,
            //                          m_min_level,
            //                          m_max_level,
            //                          min_shift,
            //                          max_shift);
            return start_shift(start_shift(i, min_shift), max_shift);
        }

        virtual int end(int, int i) const override
        {
            auto min_shift = m_min_level - m_max_level;
            auto max_shift = m_max_level - m_min_level;
            return end_shift(end_shift(i, min_shift), max_shift);
        }

        virtual int level() const override
        {
            return m_level;
        }

        int m_level;
        int m_min_level;
        int m_max_level;
    };

    struct translate_node : public node_impl
    {
        translate_node(int level, int min_level, int max_level, int t)
            : node_impl()
            , m_level(level)
            , m_min_level(min_level)
            , m_max_level(max_level)
            , m_t(t)
        {
        }

        virtual int start(int level, int s) const override
        {
            // auto min_shift = m_min_level - m_max_level;
            // auto max_shift = m_max_level - m_min_level;
            // std::cout << fmt::format(" (level: {}, m_level: {}, min_level: {}, max_level: {}, min_shift: {}, max_shift: {}, t: {}) ",
            //                          level,
            //                          m_level,
            //                          m_min_level,
            //                          m_max_level,
            //                          min_shift,
            //                          max_shift,
            //                          m_t);

            // return start_shift(start_shift(start_shift(s, min_shift), level - m_min_level) + m_t, m_max_level - level);
            return start_shift(start_shift(start_shift(s, level - m_max_level) + m_t, m_min_level - level), m_max_level - m_min_level);
        }

        virtual int end(int level, int e) const override
        {
            // auto min_shift = m_min_level - m_max_level;
            // auto max_shift = m_max_level - m_min_level;
            // return end_shift(end_shift(end_shift(e, min_shift), level - m_min_level) + m_t, m_max_level - level);
            return end_shift(end_shift(end_shift(e, level - m_max_level) + m_t, m_min_level - level), m_max_level - m_min_level);
        }

        virtual int level() const override
        {
            return m_level;
        }

        int m_level;
        int m_min_level;
        int m_max_level;
        int m_t;
    };

    struct node_t
    {
        node_t(const std::shared_ptr<node_impl>& op)
            : p_impl(op)
        {
        }

        node_t()
            : p_impl(nullptr)
        {
        }

        node_t(const node_t& rhs)
            : p_impl(rhs.p_impl)
        {
        }

        auto start(auto level, auto i) const
        {
            return p_impl->start(level, i);
        }

        auto end(auto level, auto i) const
        {
            return p_impl->end(level, i);
        }

        auto level() const
        {
            return p_impl->level();
        }

        node_t& operator=(const node_t& rhs)
        {
            p_impl = rhs.p_impl;
            return *this;
        }

        ~node_t()                                = default;
        node_t(node_t&& rhs) noexcept            = default;
        node_t& operator=(node_t&& rhs) noexcept = default;

        std::shared_ptr<node_impl> p_impl;
    };

    namespace detail

    {
        struct IntervalInfo
        {
            inline void update_node(auto& node, auto level, auto min_level, auto max_level, auto)
            {
                // if (level != min_level)
                {
                    node.push_front(node_t(std::make_shared<self_node>(level, min_level, max_level)));
                }
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

        IntervalVector(auto lca_level, auto level, auto min_level, auto max_level, iterator_t begin, iterator_t end, const std::deque<node_t>& node)
            : m_min_shift(min_level - static_cast<int>(lca_level))
            , m_max_shift(max_level - min_level)
            , m_max_level(max_level)
            , m_level(level)
            , m_shift(max_level - level)
            , m_lca_level(static_cast<int>(lca_level))
            , m_first(begin)
            , m_last(end)
            , m_current(std::numeric_limits<value_t>::min())
            , m_is_start(true)
            , m_node(node)
        {
        }

        IntervalVector()
            : m_current(sentinel<value_t>)
        {
        }

        auto start(const auto it) const
        {
            auto i = it->start << (m_max_level - m_lca_level);
            // auto i = start_shift(start_shift(it->start, m_min_shift), m_max_shift);
            // std::cout << "start: " << i;
            auto previous_level = m_lca_level;
            for (auto& n : m_node)
            {
                i = n.start(previous_level, i);
                // std::cout << " -> " << i;
                previous_level = n.level();
            }
            // std::cout << std::endl << std::endl;
            // std::cout << "dans start " << i << " " << start_shift(start_shift(i, m_min_shift), m_max_shift) << " " << m_min_shift << " "
            //           << m_max_shift << std::endl;
            // return start_shift(start_shift(i, m_min_shift), m_max_shift);
            return i;
        }

        auto end(const auto it) const
        {
            auto i = it->end << (m_max_level - m_lca_level);
            // auto i = end_shift(end_shift(it->end, m_min_shift), m_max_shift);
            auto previous_level = m_lca_level;
            for (auto& n : m_node)
            {
                i              = n.end(previous_level, i);
                previous_level = n.level();
            }
            // return end_shift(end_shift(i, m_min_shift), m_max_shift);
            return i;
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

        int m_min_shift;
        int m_max_shift;
        int m_max_level;
        int m_level;
        int m_shift;
        int m_lca_level;
        iterator_t m_first;
        iterator_t m_last;
        value_t m_current;
        bool m_is_start;
        std::deque<node_t> m_node;
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