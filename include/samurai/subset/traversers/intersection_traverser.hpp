// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <class... SetTraversers>
    class IntersectionTraverser;

    template <class... SetTraversers>
    struct SetTraverserTraits<IntersectionTraverser<SetTraversers...>>
    {
        static_assert((IsSetTraverser<SetTraversers>::value and ...));

        using FirstSetTraverser  = std::tuple_element_t<0, std::tuple<SetTraversers...>>;
        using interval_t         = typename FirstSetTraverser::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class... SetTraversers>
    class IntersectionTraverser : public SetTraverserBase<IntersectionTraverser<SetTraversers...>>
    {
        using Self = IntersectionTraverser<SetTraversers...>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS
        using Childrens = std::tuple<SetTraversers...>;

        template <size_t I>
        using IthChild = std::tuple_element<I, Childrens>::type;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

        IntersectionTraverser(const std::array<std::size_t, nIntervals>& shifts, const SetTraversers&... set_traverser)
            : m_set_traversers(set_traverser...)
            , m_shifts(shifts)
        {
            next_interval_impl();
        }

        inline bool is_empty_impl() const
        {
            return !m_current_interval.is_valid();
        }

        inline void next_interval_impl()
        {
            m_current_interval.start = 0;
            m_current_interval.end   = 0;

            while (not_is_any_child_empty() && m_current_interval.start >= m_current_interval.end)
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

        inline current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        inline bool not_is_any_child_empty() const
        {
            return std::apply(
                [](const auto&... set_traversers)
                {
                    return (!set_traversers.is_empty() and ...);
                },
                m_set_traversers);
        }

        interval_t m_current_interval;
        Childrens m_set_traversers;
        const std::array<std::size_t, nIntervals>& m_shifts;
    };

} // namespace samurai
