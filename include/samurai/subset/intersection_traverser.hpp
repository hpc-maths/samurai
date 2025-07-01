// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <SetTraverser_concept... SetTraverser>
    class IntersectionTraverser;

    template <SetTraverser_concept... SetTraverser>
        struct SetTraverserTraits < IntersectionTraverser<Op, SetTraverser>
    {
        using interval_t = typename std::tuple_element<0, SetTraverser...>::interval_t;

        static constexpr std::size_t dim = std::tuple_element<0, SetTraverser...>::dim;
    };

    template <SetTraverser_concept... SetTraversers>
    class IntersectionTraverser : public SetTraverserBase<IntersectionTraverser<SetTraversers...>>
    {
        using Self       = IntersectionTraverser<SetTraversers...>;
        using interval_t = typename SetTraverserTraits<Self>::interval_t;
        using value_t    = typename interval_t::value_t;
        using Childrens  = std::tuple<SetTraversers...>;

        template <size_t I>
        using IthChild = std::tuple_element<I, Childrens>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

      public:

        IntersectionTraverser(const std::array<std::size_t, nIntervals> shifts, const SetTraversers&... set_traversers)
            : m_set_traversers(set_traversers)
            , m_shifts(shifts)
        {
            next_interval();
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
            const auto max_start = [this]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return vmax((std::get<Is>(m_set_traversers).current_interval().start << m_shifts[Is])...);
            };
            const auto min_end = [this]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return vmin((std::get<Is>(m_set_traversers).current_interval().end << m_shifts[Is])...);
            };

            while (m_current_interval.start >= m_current_interval.end)
            {
                m_current_interval.start = max_start(std::make_index_sequence<nIntervals>{});
                m_current_interval.end   = min_end(std::make_index_sequence<nIntervals>{});

                enumerate_items(m_set_traversers,
                                [](const auto i, auto& set_traverser)
                                {
                                    if ((set_traverser.current_interval_end().end << m_shifts[i]) == m_current_interval.end)
                                    {
                                        set_traverser.next_interval();
                                    }
                                });
            }
        }

        inline interval_t& current_interval() const
        {
            return m_current_interval;
        }

      private:

        interval_t m_current_interval;
        Childrens m_set_traversers;
        std::array<std::size_t, nIntervals> m_shifts;
    };
}
