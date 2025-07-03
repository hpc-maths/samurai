// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"
#include "utils.hpp"
#include <fmt/ranges.h>

namespace samurai
{

    template <SetTraverser_concept... SetTraversers>
    class IntersectionTraverser;

    template <SetTraverser_concept... SetTraversers>
    struct SetTraverserTraits<IntersectionTraverser<SetTraversers...>>
    {
        using Childrens = std::tuple<SetTraversers...>;

        using interval_t = typename SetTraverserTraits<std::tuple_element_t<0, Childrens>>::interval_t;

        static constexpr std::size_t dim = SetTraverserTraits<std::tuple_element_t<0, Childrens>>::dim;
    };

    template <SetTraverser_concept... SetTraversers>
    class IntersectionTraverser : public SetTraverserBase<IntersectionTraverser<SetTraversers...>>
    {
        using Self       = IntersectionTraverser<SetTraversers...>;
        using interval_t = typename SetTraverserTraits<Self>::interval_t;
        using Childrens  = typename SetTraverserTraits<Self>::Childrens;
        using value_t    = typename interval_t::value_t;

        template <size_t I>
        using IthChild = std::tuple_element<I, Childrens>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

      public:

        IntersectionTraverser(const std::array<std::size_t, nIntervals> shifts, const SetTraversers&... set_traversers)
            : m_set_traversers(set_traversers...)
            , m_shifts(shifts)
        {
            next_interval();
        }

        inline bool is_empty() const
        {
            return m_current_interval.is_empty();
        }

        inline void next_interval()
        {
            m_current_interval.start = 0;
            m_current_interval.end   = 0;

            while (not_is_any_child_empty() and m_current_interval.start >= m_current_interval.end)
            {
                m_current_interval.start = std::numeric_limits<value_t>::min();
                m_current_interval.end   = std::numeric_limits<value_t>::max();

                enumerate_const_items(
                    m_set_traversers,
                    [this](const auto i, const auto& set_traverser)
                    {
                        if (!set_traverser.is_empty())
                        {
                            m_current_interval.start = std::max(m_current_interval.start,
                                                                set_traverser.current_interval().start << m_shifts[i]);
                            m_current_interval.end = std::min(m_current_interval.end, set_traverser.current_interval().end << m_shifts[i]);
                        }
                    });

                enumerate_items(
                    m_set_traversers,
                    [this](const auto i, auto& set_traverser)
                    {
                        if (!set_traverser.is_empty() && ((set_traverser.current_interval().end << m_shifts[i]) == m_current_interval.end))
                        {
                            set_traverser.next_interval();
                        }
                    });
            }
        }

        inline const interval_t& current_interval() const
        {
            return m_current_interval;
        }

      private:

        inline bool not_is_any_child_empty() const
        {
            return std::apply(
                [this](const auto&... set_traversers)
                {
                    return (!set_traversers.is_empty() and ...);
                },
                m_set_traversers);
        }

        interval_t m_current_interval;
        Childrens m_set_traversers;
        const std::array<std::size_t, nIntervals>& m_shifts;
    };
}
