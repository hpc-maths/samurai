// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <class... SetTraversers>
    class UnionTraverser;

    template <class... SetTraversers>
    struct SetTraverserTraits<UnionTraverser<SetTraversers...>>
    {
        static_assert((IsSetTraverser<SetTraversers>::value and ...));

        using FirstSetTraverser  = std::tuple_element_t<0, std::tuple<SetTraversers...>>;
        using interval_t         = typename FirstSetTraverser::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class... SetTraversers>
    class UnionTraverser : public SetTraverserBase<UnionTraverser<SetTraversers...>>
    {
        using Self = UnionTraverser<SetTraversers...>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS
        using Childrens = std::tuple<SetTraversers...>;

        template <size_t I>
        using IthChild = typename std::tuple_element<I, Childrens>::type;

        static constexpr std::size_t nIntervals = std::tuple_size<Childrens>::value;

        UnionTraverser(const std::array<std::size_t, nIntervals>& shifts, const SetTraversers&... set_traversers)
            : m_set_traversers(set_traversers...)
            , m_shifts(shifts)
        {
            next_interval_impl();
        }

        inline bool is_empty_impl() const
        {
            return m_current_interval.start == std::numeric_limits<value_t>::max();
        }

        inline void next_interval_impl()
        {
            m_current_interval.start = std::numeric_limits<value_t>::max();
            // We find the start of the interval, i.e. the smallest set_traverser.current_interval().start << m_shifts[i]
            enumerate_const_items(
                m_set_traversers,
                [this](const auto i, const auto& set_traverser)
                {
                    if (!set_traverser.is_empty() && ((set_traverser.current_interval().start << m_shifts[i]) < m_current_interval.start))
                    {
                        m_current_interval.start = set_traverser.current_interval().start << m_shifts[i];
                        m_current_interval.end   = set_traverser.current_interval().end << m_shifts[i];
                    }
                });
            // Now we find the end of the interval, i.e. the largest set_traverser.current_interval().end << m_shifts[i]
            // such that (set_traverser.current_interval().start << m_shifts[i]) < m_current_interval.end
            bool is_done = false;
            while (!is_done)
            {
                is_done = true;
                // advance set traverses that are behind current interval
                enumerate_items(
                    m_set_traversers,
                    [this](const auto i, auto& set_traverser)
                    {
                        while (!set_traverser.is_empty() && (set_traverser.current_interval().end << m_shifts[i]) <= m_current_interval.end)
                        {
                            set_traverser.next_interval();
                        }
                    });
                // try to find a new end
                enumerate_const_items(
                    m_set_traversers,
                    [&is_done, this](const auto i, const auto& set_traverser)
                    {
                        // there is an overlap
                        if (!set_traverser.is_empty() && (set_traverser.current_interval().start << m_shifts[i]) <= m_current_interval.end)
                        {
                            is_done                = false;
                            m_current_interval.end = set_traverser.current_interval().end << m_shifts[i];
                        }
                    });
            }
        }

        inline current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        interval_t m_current_interval;
        Childrens m_set_traversers;
        const std::array<std::size_t, nIntervals>& m_shifts;
    };

} // namespace samurai
