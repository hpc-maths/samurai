// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <SetTraverser_concept... SetTraverser>
    class UnionTraverser;

    template <SetTraverser_concept... SetTraversers>
    struct SetTraverserTraits<UnionTraverser<SetTraversers...>>
    {
        using Childrens = std::tuple<SetTraversers...>;

        using interval_t = typename SetTraverserTraits<std::tuple_element_t<0, Childrens>>::interval_t;

        static constexpr std::size_t dim = SetTraverserTraits<std::tuple_element_t<0, Childrens>>::dim;
    };

    template <SetTraverser_concept... SetTraversers>
    class UnionTraverser : public SetTraverserBase<UnionTraverser<SetTraversers...>>
    {
        using Self       = UnionTraverser<SetTraversers...>;
        using interval_t = typename SetTraverserTraits<Self>::interval_t;
        using Childrens  = typename SetTraverserTraits<Self>::Childrens;
        using value_t    = typename interval_t::value_t;

        template <size_t I>
        using IthChild = std::tuple_element<I, Childrens>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

      public:

        UnionTraverser(const std::array<std::size_t, nIntervals>& shifts, const SetTraversers&... set_traverser)
            : m_set_traversers(set_traverser...)
            , m_shifts(shifts)
        {
            compute_current_interval();
        }

        inline bool is_empty() const
        {
            return std::apply(
                [this](const auto&... set_traversers)
                {
                    return (set_traversers.is_empty() && ...);
                },
                m_set_traversers);
        }

        inline void next_interval()
        {
            // we want all of our child to advance until they have not overlap with m_current_interval.
            enumerate_items(
                m_set_traversers,
                [this](const auto i, auto& set_traverser)
                {
                    while (!set_traverser.is_empty() && (set_traverser.current_interval().start << m_shifts[i]) <= m_current_interval.end)
                    {
                        set_traverser.next_interval();
                    }
                });
            // we have passed, m_current_interval, now re-compute it.
            compute_current_interval();
        }

        inline const interval_t& current_interval() const
        {
            return m_current_interval;
        }

      private:

        void compute_current_interval()
        {
            // first we compute the first interval
            m_current_interval.start = std::numeric_limits<value_t>::max();
            enumerate_const_items(
                m_set_traversers,
                [this](const auto i, const auto& set_traverser)
                {
                    if ((!set_traverser.is_empty()) and (set_traverser.current_interval().start << m_shifts[i]) < m_current_interval.start)
                    {
                        m_current_interval.start = set_traverser.current_interval().start << m_shifts[i];
                        m_current_interval.end   = set_traverser.current_interval().end << m_shifts[i];
                    }
                });
            enumerate_const_items(
                m_set_traversers,
                [this](const auto i, const auto& set_traverser)
                {
                    // there is an overlap
                    if ((!set_traverser.is_empty()) && (set_traverser.current_interval().start << m_shifts[i]) <= m_current_interval.end)
                    {
                        m_current_interval.start = std::min(m_current_interval.start,
                                                            (set_traverser.current_interval().start << m_shifts[i]));
                        m_current_interval.end   = std::max(m_current_interval.end, (set_traverser.current_interval().end << m_shifts[i]));
                    }
                });
        }

        interval_t m_current_interval;
        Childrens m_set_traversers;
        const std::array<std::size_t, nIntervals>& m_shifts;
    };

}
