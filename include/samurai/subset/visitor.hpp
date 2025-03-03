// Copyright 2018-2024 the samurai's authors
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
            , m_start(start)
            , m_end(end)
            , m_work(data)
            , m_take_data(true)
        {
        }

        IntervalListRange(const container_t& data, const container_t& w)
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
    class IntervalListVisitor
    {
      public:

        using container_t = container_;
        using base_t      = IntervalListRange<container_t>;
        using iterator_t  = typename base_t::iterator_t;
        using interval_t  = typename base_t::value_t;
        using value_t     = typename interval_t::value_t;

        IntervalListVisitor(auto lca_level, auto level, auto max_level, IntervalListRange<container_t>&& intervals)
            : m_lca_level(static_cast<int>(lca_level))
            , m_shift2dest(static_cast<int>(max_level) - static_cast<int>(level))
            , m_shift2ref(static_cast<int>(max_level) - static_cast<int>(lca_level))
            , m_intervals(std::move(intervals))
            , m_first(m_intervals.begin())
            , m_last(m_intervals.end())
            , m_current(std::numeric_limits<value_t>::min())
            , m_is_start(true)
        {
        }

        explicit IntervalListVisitor(IntervalListRange<container_t>&& intervals)
            : m_lca_level(std::numeric_limits<std::size_t>::infinity())
            , m_shift2dest(std::numeric_limits<std::size_t>::infinity())
            , m_shift2ref(std::numeric_limits<std::size_t>::infinity())
            , m_intervals(std::move(intervals))
            , m_first(m_intervals.begin())
            , m_last(m_intervals.end())
            , m_current(sentinel<value_t>)
            , m_is_start(true)
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
            // Recall that we check if scan is inside an interval defined as [start,
            // end[. The end of the interval is not included.
            //
            // if the m_current value is the start of the interval which means m_is_start =
            // true then if scan is lower than m_current, scan is not in the
            // interval.
            //
            // if the m_current value is the end of the interval which means m_is_start = false
            // then if scan is lower than m_current, scan is in the interval.
            return m_current != sentinel<value_t> && !((scan < m_current) ^ (!m_is_start));
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
            auto& [start_fct, end_fct] = start_and_stop; // cppcheck-suppress variableScope
            if (m_current == std::numeric_limits<value_t>::min())
            {
                m_current = start(m_first, start_fct);
                return;
            }

            if (m_current == scan)
            {
                if (m_is_start)
                {
                    m_current = end(m_first, end_fct);
                    while (m_first + 1 != m_last && m_current >= start(m_first + 1, start_fct))
                    {
                        ++m_first;
                        m_current = end(m_first, end_fct);
                    }
                }
                else
                {
                    ++m_first;

                    if (m_first == m_last)
                    {
                        m_current = sentinel<value_t>;
                        return;
                    }
                    m_current = start(m_first, start_fct);
                }
                m_is_start = !m_is_start;
            }
        }

      private:

        int m_lca_level;
        int m_shift2dest;
        int m_shift2ref;
        IntervalListRange<container_t> m_intervals;
        iterator_t m_first;
        iterator_t m_last;
        value_t m_current;
        bool m_is_start;
    };

    template <class Operator, class... S>
    class SetTraverser
    {
      public:

        static constexpr std::size_t dim = get_set_dim_v<S...>;
        using set_type                   = std::tuple<S...>;
        using interval_t                 = get_interval_t<S...>;

        SetTraverser(int shift, const Operator& op, S&&... s)
            : m_shift(shift)
            , m_operator(op)
            , m_s(std::forward<S>(s)...)
        {
        }

        inline auto shift() const
        {
            return m_shift;
        }

        inline bool is_in(auto scan) const
        {
            return std::apply(
                [this, scan](auto&&... args)
                {
                    return m_operator.is_in(scan, args...);
                },
                m_s);
        }

        inline bool is_empty() const
        {
            return std::apply(
                [this](auto&&... args)
                {
                    return m_operator.is_empty(args...);
                },
                m_s);
        }

        inline auto min() const
        {
            return std::apply(
                [](auto&&... args)
                {
                    return compute_min(args.min()...);
                },
                m_s);
        }

        template <class StartEnd>
        void next(auto scan, StartEnd&& start_and_stop)
        {
            zip_apply(
                [scan](auto& arg, auto& start_end_fct)
                {
                    arg.next(scan, start_end_fct);
                },
                m_s,
                std::forward<StartEnd>(start_and_stop));
        }

      private:

        int m_shift;
        Operator m_operator;
        set_type m_s;
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
