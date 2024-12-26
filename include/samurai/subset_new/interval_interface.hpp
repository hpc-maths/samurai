#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <xtl/xiterator_base.hpp>

namespace samurai::experimental
{
    static constexpr int sentinel = std::numeric_limits<int>::max();

    namespace detail
    {
        template <class T>
        inline T end_shift(T value, T shift)
        {
            return shift >= 0 ? value << shift : ((value - 1) >> -shift) + 1;
        }

        template <class T>
        inline T start_shift(T value, T shift)
        {
            return shift >= 0 ? value << shift : value >> -shift;
        }

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
        using value_t    = typename iterator_t::value_type::value_t;

        IntervalVector(auto lca_level, auto level, auto min_level, auto max_level, iterator_t begin, iterator_t end)
            : m_min_shift(min_level - static_cast<int>(lca_level))
            , m_max_shift(max_level - min_level)
            , m_shift(max_level - level)
            , m_lca_level(static_cast<int>(lca_level))
            , m_first(begin)
            , m_last(end)
            , m_current(std::numeric_limits<int>::min())
            , m_is_start(true)
        {
        }

        auto start(const auto i) const
        {
            return detail::start_shift(detail::start_shift(i, m_min_shift), m_max_shift);
        }

        auto end(const auto i) const
        {
            return detail::end_shift(detail::end_shift(i, m_min_shift), m_max_shift);
        }

        bool is_in(auto scan) const
        {
            return m_current != sentinel && !((scan < m_current) ^ !m_is_start);
        }

        bool is_empty() const
        {
            return m_current == sentinel;
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
            if (m_current == std::numeric_limits<int>::min())
            {
                m_current = start(iop.start_op(m_lca_level, m_first));
                return;
            }

            if (m_current == scan)
            {
                if (m_is_start)
                {
                    // std::cout << "end of m_first: " << end_op(m_first) << std::endl;
                    m_current = end(iop.end_op(m_lca_level, m_first));
                    while (m_first + 1 != m_last && m_current >= start(iop.start_op(m_lca_level, m_first + 1)))
                    {
                        m_first++;
                        m_current = end(iop.end_op(m_lca_level, m_first));
                        // std::cout << "update end in while loop: " << m_current << std::endl;
                    }
                    // std::cout << "update end: " << m_current << std::endl;
                }
                else
                {
                    m_first++;
                    if (m_first == m_last)
                    {
                        m_current = sentinel;
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

    template <class const_iterator_t, class offset_t>
    class offset_iterator
    {
      public:

        using iterator_category = std::forward_iterator_tag;
        using const_iterator    = const_iterator_t;
        using value_type        = typename const_iterator_t::value_type;
        using const_reference   = typename const_iterator_t::reference;
        using pointer           = typename const_iterator_t::pointer;
        using const_pointer     = const pointer;

        offset_iterator(const_iterator_t interval_it, const offset_t& offset, bool is_end = false)
            : m_interval_it(interval_it)
            , m_interval_end(interval_it + (!is_end ? static_cast<std::ptrdiff_t>(offset.back()) : 0))
            , m_offset(offset)
            , m_count(1)
        {
            p_first.reserve(offset.size() - 1);
            p_last.reserve(offset.size() - 1);
            for (std::size_t i = 0; i < offset.size() - 1; ++i)
            {
                if (is_end)
                {
                    p_first.push_back(interval_it + static_cast<std::ptrdiff_t>(offset[i + 1]));
                }
                else
                {
                    p_first.push_back(interval_it + static_cast<std::ptrdiff_t>(offset[i]));
                }
                p_last.push_back(interval_it + static_cast<std::ptrdiff_t>(offset[i + 1]));
            }

            next();
        }

        void next()
        {
            using value_t = typename value_type::value_t;

            update_count();

            if (m_count == m_offset.size())
            {
                m_current = {0, 0};
                return;
            }

            auto start = std::numeric_limits<value_t>::max();
            auto end   = std::numeric_limits<value_t>::min();

            for (std::size_t i = 0; i < p_first.size(); ++i)
            {
                if (p_first[i] != p_last[i])
                {
                    if (start >= p_first[i]->start)
                    {
                        start         = p_first[i]->start;
                        end           = std::max(end, p_first[i]->end);
                        m_interval_it = p_first[i];
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

        offset_iterator& operator++()
        {
            next();
            if (!m_current.is_valid())
            {
                m_interval_it = m_interval_end;
            }
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
            return m_interval_it == other.m_interval_it;
        }

        bool operator!=(const offset_iterator& other) const
        {
            return !(*this == other);
        }

      private:

        void update_count()
        {
            if (m_count != m_offset.size())
            {
                m_count = 1;
                for (std::size_t i = 0; i < p_first.size(); ++i)
                {
                    if (p_first[i] == p_last[i])
                    {
                        m_count++;
                    }
                }
            }
        }

        value_type m_current;
        std::vector<const_iterator_t> p_first;
        std::vector<const_iterator_t> p_last;
        const_iterator_t m_interval_it;
        const_iterator_t m_interval_end;
        const offset_t& m_offset;
        std::size_t m_count;
    };

    template <class const_iterator_t, class offset_t>
    auto operator+(const offset_iterator<const_iterator_t, offset_t>& it, int i)
    {
        offset_iterator<const_iterator_t, offset_t> temp = it;
        for (int ii = 0; ii < i; ++ii)
        {
            temp++;
        }
        return temp;
    }

    auto IntervalVectorOffset(auto lca_level, auto level, auto min_level, auto max_level, const auto& x, const auto& offset)
    {
        return IntervalVector(lca_level,
                              level,
                              min_level,
                              max_level,
                              offset_iterator(x.cbegin(), offset),
                              offset_iterator(x.cbegin() + static_cast<std::ptrdiff_t>(offset.back()), offset, true));
    }
}