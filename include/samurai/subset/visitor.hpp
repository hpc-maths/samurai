// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <limits>

#include "utils.hpp"

namespace samurai
{
    template <class container_>
    class IntervalListRange
    {
      public:

        using container_t = container_;
        using value_t     = typename container_t::value_type;
        using iterator_t  = typename container_t::const_iterator;

        IntervalListRange(const container_t& data, std::ptrdiff_t start, std::ptrdiff_t end)
            : m_data(data)
            , m_work(data)
            , m_begin(data.cbegin() + start)
            , m_end(data.cbegin() + end)
        {
        }

        IntervalListRange(const container_t& data, const container_t& w)
            : m_data(data)
            , m_work(w)
            , m_begin(w.cbegin())
            , m_end(w.cend())
        {
        }

        inline auto begin() const
        {
            return m_begin;
        }

        inline auto end() const
        {
            return m_end;
        }

      private:

        const container_t& m_data;
        const container_t& m_work;
        iterator_t m_begin;
        iterator_t m_end;
    };

    template <class container_>
    class IntervalListVisitor
    {
      public:

        using container_t = container_;
        using base_t      = IntervalListRange<container_t>;
        using iterator_t  = typename base_t::iterator_t;
        using interval_t  = typename base_t::value_t;
        using value_t     = typename interval_t::value_t;

        IntervalListVisitor(auto lca_level, auto level, auto min_level, auto max_level, const IntervalListRange<container_t>& intervals)
            : m_shift2min(static_cast<int>(lca_level) - static_cast<int>(min_level))
            , m_shift2dest(static_cast<int>(max_level) - static_cast<int>(level))
            , m_shift2ref(static_cast<int>(max_level) - static_cast<int>(min_level))
            , m_intervals(intervals)
            , m_first(intervals.begin())
            , m_last(intervals.end())
            , m_current(std::numeric_limits<value_t>::min())
            , m_is_start(true)
            , m_unvalid(false)
        {
        }

        explicit IntervalListVisitor(IntervalListRange<container_t>&& intervals)
            // : m_shift2min(std::numeric_limits<std::size_t>::infinity())
            // , m_shift2dest(std::numeric_limits<std::size_t>::infinity())
            // , m_shift2ref(std::numeric_limits<std::size_t>::infinity())

            : m_intervals(std::move(intervals))
            // , m_first(m_intervals.begin())
            // , m_last(m_intervals.end())
            , m_current(sentinel<value_t>)
            // , m_is_start(true)
            , m_unvalid(true)
        {
        }

        inline auto start(const auto& it) const
        {
            return (it->start >> m_shift2min) << m_shift2ref;
        }

        inline auto end(const auto& it) const
        {
            return (((it->end - 1) >> m_shift2min) + 1) << m_shift2ref;
        }

        inline bool is_in(auto scan) const
        {
            // Recall that we check if scan is inside an interval defined as [start,
            // end[. The end of the interval is not included.
            //
            // if the m_current value is the start of the interval which means m_is_start =
            // true then if scan is lower than m_current, scan is not in the
            // interval.
            //
            // if the m_current value is the end of the interval which means m_is_start = false
            // then if scan is lower than m_current, scan is in the interval.
            return m_current != sentinel<value_t> && m_current_interval.contains(scan); // !((scan < m_current) ^ (!m_is_start));
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

        inline void next_interval()
        {
            if (m_current != std::numeric_limits<value_t>::min())
            {
                ++m_first;
            }

            if (m_first == m_last)
            {
                m_current = sentinel<value_t>;
                return;
            }

            auto i_start = start(m_first);
            auto i_end   = end(m_first);
            while (m_first + 1 != m_last && i_end >= start(m_first + 1))
            {
                ++m_first;
                i_end = end(m_first);
            }
            m_current_interval = {i_start, i_end};

            if (m_current_interval.is_valid())
            {
                m_current = m_current_interval.start;
            }
            else
            {
                m_current = sentinel<value_t>;
            }
            // std::cout << "next interval: " << m_current_interval << std::endl;
        }

        inline void next(auto scan)
        {
            if (m_unvalid)
            {
                return;
            }

            if (m_current == std::numeric_limits<value_t>::min())
            {
                next_interval();
                return;
            }

            if (m_current == scan)
            {
                if (m_is_start)
                {
                    m_current = m_current_interval.end;
                }
                else
                {
                    next_interval();
                }
                m_is_start = !m_is_start;
            }
        }

        inline auto& current_interval()
        {
            return m_current_interval;
        }

      private:

        int m_shift2min;
        int m_shift2dest;
        int m_shift2ref;
        IntervalListRange<container_t> m_intervals;
        iterator_t m_first;
        iterator_t m_last;
        value_t m_current;
        interval_t m_current_interval;
        bool m_is_start;
        bool m_unvalid;
    };

    template <class Operator, class StartAndStopOp, class... S>
    class SetTraverser
    {
      public:

        static constexpr std::size_t dim = get_set_dim_v<S...>;
        using set_type                   = std::tuple<S...>;
        using interval_t                 = get_interval_t<S...>;
        using value_t                    = typename interval_t::value_t;

        template <class... ST>
        SetTraverser(int shift, const Operator& op, const StartAndStopOp& start_and_stop_op, S&&... s)
            : m_shift(shift)
            , m_operator(op)
            , m_start_and_stop_op(start_and_stop_op)
            , m_s(std::forward<S>(s)...)
            , m_current(std::numeric_limits<value_t>::min())
            , m_next_interval({std::numeric_limits<value_t>::min(), std::numeric_limits<value_t>::min()})
            , m_is_start(true)
        {
        }

        inline auto shift() const
        {
            return m_shift;
        }

        inline bool is_in(auto scan) const
        {
            return m_current != sentinel<value_t> && m_current_interval.contains(scan); //!((scan < m_current) ^ (!m_is_start));
        }

        inline bool is_empty() const
        {
            return m_current == sentinel<value_t>;
        }

        inline auto min() const
        {
            return m_current;
        }

        inline void next(auto scan)
        {
            if (m_current == std::numeric_limits<value_t>::min())
            {
                m_scan = child_min();
                next_interval();
                return;
            }

            if (m_current == scan)
            {
                if (m_is_start)
                {
                    m_current = m_current_interval.end;
                }
                else
                {
                    next_interval();
                }
                m_is_start = !m_is_start;
            }
        }

        inline bool child_is_in(auto scan) const
        {
            return std::apply(
                [this, scan](auto&... args)
                {
                    return m_operator.is_in(scan, args...);
                },
                m_s);
        }

        inline auto child_min() const
        {
            return std::apply(
                [](auto&... args)
                {
                    return compute_min(args.min()...);
                },
                m_s);
        }

        inline void child_next(auto scan)
        {
            std::apply(
                [scan](auto&... arg)
                {
                    (arg.next(scan), ...);
                },
                m_s);
        }

        inline bool child_is_empty() const
        {
            return std::apply(
                [this](auto&... args)
                {
                    return m_operator.is_empty(args...);
                },
                m_s);
        }

        void find_next()
        {
            if (!child_is_empty())
            {
                if constexpr (sizeof...(S) == 1)
                {
                    std::get<0>(m_s).next_interval();
                    m_next_interval = std::get<0>(m_s).current_interval();
                    return;
                }
                else
                {
                    int r_ipos = 0;

                    child_next(m_scan);
                    m_scan = child_min();

                    // std::cout << "SetTraverser: find_next with scan = " << m_scan << std::endl;
                    while (m_scan < sentinel<value_t> && !child_is_empty())
                    {
                        bool is_in = child_is_in(m_scan);

                        if (is_in && r_ipos == 0)
                        {
                            m_next_interval.start = m_scan;
                            r_ipos                = 1;
                        }
                        else if (!is_in && r_ipos == 1)
                        {
                            m_next_interval.end = m_scan;
                            r_ipos              = 0;
                            return;
                        }

                        child_next(m_scan);
                        m_scan = child_min();
                    }
                }
            }
            // std::cout << "SetTraverser: find_next finished with scan = " << m_scan << std::endl;
            m_next_interval = {0, 0};
        }

        inline void next_interval()
        {
            if (m_current == sentinel<value_t>)
            {
                return;
            }

            if (m_next_interval.start == std::numeric_limits<value_t>::min())
            {
                find_next();
            }
            // std::cout << "SetTraverser: next_interval with itmp = " << itmp << std::endl;
            if (!m_next_interval.is_valid())
            {
                m_current = sentinel<value_t>;
                // m_current_interval = {0, 0};
                return;
            }

            m_current_interval = m_next_interval;
            m_start_and_stop_op(m_current_interval);

            // std::cout << "SetTraverser: next_interval with current interval = " << m_current_interval << std::endl;
            find_next();
            // std::cout << "SetTraverser: next_interval after find_next with itmp = " << itmp << std::endl;
            while (m_next_interval.is_valid() && m_current_interval.end >= m_start_and_stop_op.start(m_next_interval))
            {
                m_current_interval.end = m_start_and_stop_op.end(m_next_interval);
                find_next();
            }
            if (m_current_interval.is_valid())
            {
                m_current = m_current_interval.start;
            }
            else
            {
                m_current = sentinel<value_t>;
            }
            // std::cout << "next interval in SetTraverser: " << m_current_interval << std::endl;
        }

        inline auto& current_interval()
        {
            return m_current_interval;
        }

      private:

        int m_shift;
        Operator m_operator;
        StartAndStopOp m_start_and_stop_op;
        set_type m_s;
        value_t m_current;
        interval_t m_current_interval;
        interval_t m_next_interval;
        bool m_is_start;
        value_t m_scan;
    };

    struct IntersectionOp
    {
        bool is_in(auto scan, const auto&... args) const
        {
            return (args.is_in(scan) && ...);
        }

        bool is_empty(const auto&... args) const
        {
            return (args.is_empty() || ...);
        }

        bool exist(const auto&... args) const
        {
            return (args.exist() && ...);
        }
    };

    struct UnionOp
    {
        bool is_in(auto scan, const auto&... args) const
        {
            return (args.is_in(scan) || ...);
        }

        bool is_empty(const auto&... args) const
        {
            return (args.is_empty() && ...);
        }

        bool exist(const auto&... args) const
        {
            return (args.exist() || ...);
        }
    };

    struct DifferenceOp
    {
        bool is_in(auto scan, const auto& arg, const auto&... args) const
        {
            return arg.is_in(scan) && !(args.is_in(scan) || ...);
        }

        bool is_empty(const auto& arg, const auto&...) const
        {
            return arg.is_empty();
        }

        bool exist(const auto& arg, const auto&...) const
        {
            return arg.exist();
        }
    };

    struct Difference2Op
    {
        bool is_in(auto scan, const auto& arg, const auto&...) const
        {
            return arg.is_in(scan);
        }

        bool is_empty(const auto& arg, const auto&...) const
        {
            return arg.is_empty();
        }

        bool exist(const auto& arg, const auto&...) const
        {
            return arg.exist();
        }
    };

    template <std::size_t d, class operator_t>
    auto get_operator(const operator_t& op)
    {
        return op;
    }

    template <std::size_t d>
    auto get_operator(const DifferenceOp& op)
    {
        if constexpr (d == 1)
        {
            return op;
        }
        else
        {
            return Difference2Op();
        }
    }

    struct SelfOp
    {
        bool is_in(auto scan, const auto& arg) const
        {
            return arg.is_in(scan);
        }

        bool is_empty(const auto& arg) const
        {
            return arg.is_empty();
        }

        bool exist(const auto& arg) const
        {
            return arg.exist();
        }
    };
}
