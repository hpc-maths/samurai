// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <SetTraverser_concept... SetTraverser>
    class UnionTraverser;

    template <SetTraverser_concept... SetTraverser>
        struct SetTraverserTraits < UnionTraverser<Op, SetTraverser>
    {
        using interval_t = typename std::tuple_element<0, SetTraverser...>::interval_t;

        static constexpr std::size_t dim = std::tuple_element<0, SetTraverser...>::dim;
    };

    template <SetTraverser_concept... SetTraversers>
    class UnionTraverser : public SetTraverserBase<UnionTraverser<SetTraversers...>>
    {
        using Self       = UnionTraverser<Op, SetTraversers...>;
        using interval_t = typename SetTraverserTraits<Self>::interval_t;
        using value_t    = typename interval_t::value_t;
        using Childrens  = std::tuple<SetTraversers...>;

        template <size_t I>
        using IthChild = std::tuple_element<I, Childrens>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

      public:

        UnionTraverser(const SetTraversers&... set_traverser)
            : m_set_traversers(set_traverser)
        {
            compute_current_interval();
            Base::init_current();
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
                    static_for<0, nIntervals>::apply([this](const auto i)
                    {
                IthChild<i>& set_traverser = std::get<i>(m_set_traversers);
                while ((!set_traverser.is_empty()) && set_traverser.current_interval().start <= m_current_interval.end)
                {
                    set_traverser.next_interval();
                }
                    };
                    // we have passed, m_current_interval, now re-compute it.
                    compute_current_interval();
        }

        inline interval_t& current_interval()
        {
            return m_current_interval;
        }

      private:

        void compute_current_interval()
        {
            // first we compute the first interval
            m_current_interval.start = std::apply(
                [](const auto&... traversers) -> interval_t
                {
                    compute_min(traversers.current_interval()...);
                },
                m_set_traversers);

           static_for<0, nIntervals>::apply([this](const auto i)
                        {
                const IthChild<i>& set_traverser = std::get<i>(m_set_traversers);
                if ((!set_traverser.is_empty()) && set_traverser.current_interval().start <= m_current_interval.end) // there is an overlap
                {
                    m_current_interval.start = std::min(m_current_interval.start, set_traverser.current_interval().start);
                    m_current_interval.end   = std::min(m_current_interval.end, set_traverser.current_interval().end);
                }
                        };
        }

        interval_t m_current_interval;
        Childrens m_set_traversers;
    };

}
