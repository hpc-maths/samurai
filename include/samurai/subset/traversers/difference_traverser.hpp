// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../static_algorithm.hpp"
#include "set_traverser_base.hpp"

namespace samurai
{

    template <class... SetTraversers>
    class DifferenceTraverser;

    template <class... SetTraversers>
    struct SetTraverserTraits<DifferenceTraverser<SetTraversers...>>
    {
        static_assert((IsSetTraverser<SetTraversers>::value and ...));

        using FirstSetTraverser  = std::tuple_element_t<0, std::tuple<SetTraversers...>>;
        using interval_t         = typename FirstSetTraverser::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class... SetTraversers>
    class DifferenceTraverser : public SetTraverserBase<DifferenceTraverser<SetTraversers...>>
    {
        using Self = DifferenceTraverser<SetTraversers...>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS
        using Childrens = std::tuple<SetTraversers...>;

        template <size_t I>
        using IthChild = typename std::tuple_element<I, Childrens>::type;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

        DifferenceTraverser(const std::array<std::size_t, nIntervals>& shifts, const SetTraversers&... set_traversers)
            : m_min_start(std::numeric_limits<value_t>::min())
            , m_set_traversers(set_traversers...)
            , m_shifts(shifts)
        {
            compute_current_interval();
        }

        inline bool is_empty_impl() const
        {
            return std::get<0>(m_set_traversers).is_empty();
        }

        inline void next_interval_impl()
        {
            advance_ref_interval();
            compute_current_interval();
        }

        inline current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        inline void advance_ref_interval()
        {
            if (m_current_interval.end != std::get<0>(m_set_traversers).current_interval().end << m_shifts[0])
            {
                // we have removed the beginning of the current interval.
                // so ve remove [m_current_interval.start, m_current_interval.end) from std::get<0>(m_set_traversers).current_interval()
                m_min_start = m_current_interval.end;
            }
            else
            {
                // all of the current interval has been removed.
                // move to the next one.
                std::get<0>(m_set_traversers).next_interval();
            }
        }

        inline void compute_current_interval()
        {
            while (!std::get<0>(m_set_traversers).is_empty() && !try_to_compute_current_interval())
            {
                advance_ref_interval();
            }
        }

        inline bool try_to_compute_current_interval()
        {
            assert(!std::get<0>(m_set_traversers).is_empty());

            m_current_interval.start = std::max(m_min_start, std::get<0>(m_set_traversers).current_interval().start << m_shifts[0]);
            m_current_interval.end   = std::get<0>(m_set_traversers).current_interval().end << m_shifts[0];

            static_for<1, nIntervals>::apply(
                [this](const auto i)
                {
                    IthChild<i>& set_traverser = std::get<i>(m_set_traversers);

                    while (!set_traverser.is_empty() && (set_traverser.current_interval().end << m_shifts[i]) < m_current_interval.start)
                    {
                        set_traverser.next_interval();
                    }

                    if (!set_traverser.is_empty() && (set_traverser.current_interval().start << m_shifts[i]) <= m_current_interval.start)
                    {
                        m_current_interval.start = set_traverser.current_interval().end << m_shifts[i];
                        m_min_start              = m_current_interval.start;
                        set_traverser.next_interval();
                    }
                    if (!set_traverser.is_empty() && (set_traverser.current_interval().start << m_shifts[i]) <= m_current_interval.end)
                    {
                        m_current_interval.end = set_traverser.current_interval().start << m_shifts[i];
                    }
                });

            return m_current_interval.is_valid();
        }

        interval_t m_current_interval;
        value_t m_min_start;
        Childrens m_set_traversers;
        const std::array<std::size_t, nIntervals>& m_shifts;
    };

} // namespace samurai
