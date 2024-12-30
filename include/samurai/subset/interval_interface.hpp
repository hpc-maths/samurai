// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
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
        node_impl()                                      = default;
        virtual int start(int level, int i) const        = 0;
        virtual int end(int level, int i) const          = 0;
        virtual std::unique_ptr<node_impl> clone() const = 0;

        virtual ~node_impl() = default;

        node_impl(const node_impl&)            = delete;
        node_impl& operator=(const node_impl&) = delete;
        node_impl(node_impl&&)                 = delete;
        node_impl& operator=(node_impl&&)      = delete;
    };

    struct final_node : public node_impl
    {
        final_node()
            : node_impl()
        {
        }

        virtual int start(int, int i) const override
        {
            return i;
        }

        virtual int end(int, int i) const override
        {
            return i;
        }

        std::unique_ptr<node_impl> clone() const override
        {
            return std::make_unique<final_node>();
        }
    };

    struct self_node : public node_impl
    {
        self_node()
            : node_impl()
            , m_op(std::make_unique<final_node>())
        {
        }

        self_node(const std::unique_ptr<node_impl>& op)
            : node_impl()
            , m_op(op->clone())
        {
        }

        virtual int start(int level, int i) const override
        {
            return m_op->start(level, i);
        }

        virtual int end(int level, int i) const override
        {
            return m_op->end(level, i);
        }

        std::unique_ptr<node_impl> clone() const override
        {
            return std::make_unique<self_node>(m_op);
        }

        std::unique_ptr<node_impl> m_op;
    };

    struct translate_node : public node_impl
    {
        translate_node(const std::unique_ptr<node_impl>& op, int level, int t)
            : node_impl()
            , m_op(op->clone())
            , m_level(level)
            , m_t(t)
        {
        }

        translate_node(int level, int t)
            : node_impl()
            , m_op(std::make_unique<final_node>())
            , m_level(level)
            , m_t(t)
        {
        }

        virtual int start(int level, int i) const override
        {
            return m_op->start(level, i) + start_shift(m_t, level - m_level);
        }

        virtual int end(int level, int i) const override
        {
            return m_op->end(level, i) + end_shift(m_t, level - m_level);
        }

        virtual std::unique_ptr<node_impl> clone() const override
        {
            return std::make_unique<translate_node>(m_op, m_level, m_t);
        }

        std::unique_ptr<node_impl> m_op;
        int m_level;
        int m_t;
    };

    struct node_t
    {
        node_t(std::unique_ptr<node_impl>&& op)
            : p_impl(std::move(op))
        {
        }

        node_t()
            : p_impl(std::make_unique<final_node>())
        {
        }

        node_t(const node_t& rhs)
            : p_impl(rhs.p_impl->clone())
        {
        }

        auto start(auto level, auto i)
        {
            return p_impl->start(level, i);
        }

        auto end(auto level, auto i)
        {
            return p_impl->end(level, i);
        }

        node_t& operator=(const node_t& rhs)
        {
            p_impl.reset();
            p_impl = rhs.p_impl->clone();
            return *this;
        }

        ~node_t()                                = default;
        node_t(node_t&& rhs) noexcept            = default;
        node_t& operator=(node_t&& rhs) noexcept = default;

        std::unique_ptr<node_impl> p_impl;
    };

    namespace detail

    {
        struct IntervalInfo
        {
            inline node_t get_node(auto, auto)
            {
                return node_t();
            }

            inline auto get_node(const node_t& node, auto, auto)
            {
                return node_t(std::make_unique<self_node>(node.p_impl));
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

        IntervalVector(auto lca_level, auto level, auto min_level, auto max_level, iterator_t begin, iterator_t end, const node_t& node)
            : m_min_shift(min_level - static_cast<int>(lca_level))
            , m_max_shift(max_level - min_level)
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

        void next(auto scan)
        {
            // std::cout << std::endl;
            // std::cout << "m_current in next: " << m_current << " " << std::numeric_limits<value_t>::min() << std::endl;
            if (m_current == std::numeric_limits<value_t>::min())
            {
                m_current = start(m_node.start(m_lca_level, m_first->start));
                // std::cout << "first m_current: " << m_current << std::endl;
                return;
            }

            if (m_current == scan)
            {
                if (m_is_start)
                {
                    m_current = end(m_node.end(m_lca_level, m_first->end));
                    // std::cout << "change m_current: " << m_current << std::endl;
                    while (m_first + 1 != m_last && m_current >= start(m_node.start(m_lca_level, (m_first + 1)->start)))
                    {
                        m_first++;
                        m_current = end(m_node.end(m_lca_level, m_first->end));
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
                    m_current = start(m_node.start(m_lca_level, m_first->start));
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
        node_t m_node;
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