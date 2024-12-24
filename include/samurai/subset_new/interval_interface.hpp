#pragma once

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

    } // namespace detail

    template <class vInterval_t>
    struct IntervalVector
    {
        using iterator_t       = typename vInterval_t::iterator;
        using const_iterator_t = typename vInterval_t::const_iterator;
        using value_t          = typename vInterval_t::value_type::value_t;

        IntervalVector(auto lca_level, auto level, auto min_level, auto max_level, iterator_t begin, iterator_t end)
            : m_min_shift(min_level - lca_level)
            , m_max_shift(max_level - min_level)
            , m_shift(max_level - level)
            , m_first(begin)
            , m_last(end)
            , m_current(start(m_first))
            , m_is_start(true)
        {
            // std::cout << "start with: " << m_current << " " << level << " " << m_min_shift << " " << m_max_shift << std::endl;
        }

        auto start_op(const auto it) const
        {
            return it->start;
        }

        auto start(const auto it) const
        {
            return detail::start_shift(detail::start_shift(start_op(it), m_min_shift), m_max_shift);
        }

        auto end_op(const auto it) const
        {
            return it->end;
        }

        auto end(const auto it) const
        {
            return detail::end_shift(detail::end_shift(end_op(it), m_min_shift), m_max_shift);
        }

        bool is_in(auto scan) const

        {
            return m_current != sentinel && !((scan < m_current) ^ !m_is_start);
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
            if (m_current == scan)
            {
                if (m_is_start)
                {
                    // std::cout << "end of m_first: " << end_op(m_first) << std::endl;
                    m_current = end(m_first);
                    while (m_first + 1 != m_last && m_current >= start(m_first + 1))
                    {
                        m_first++;
                        m_current = end(m_first);
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
                    m_current = start(m_first);
                    // std::cout << "update start: " << m_current << std::endl;
                }
                m_is_start = !m_is_start;
            }
        }

        int m_min_shift;
        int m_max_shift;
        int m_shift;
        iterator_t m_first;
        iterator_t m_last;
        value_t m_current;
        bool m_is_start;
    };
}