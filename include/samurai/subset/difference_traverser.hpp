// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../static_algorithm.hpp"
#include "set_traverser_base.hpp"
#include "utils.hpp"

namespace samurai
{

    template <SetTraverser_concept... SetTraversers>
    class DifferenceTraverser;

    template <SetTraverser_concept... SetTraversers>
    struct SetTraverserTraits<DifferenceTraverser<SetTraversers...>>
    {
        using Childrens          = std::tuple<SetTraversers...>;
        using interval_t         = typename SetTraverserTraits<std::tuple_element_t<0, Childrens>>::interval_t;
        using current_interval_t = const interval_t&;

        static constexpr std::size_t dim = SetTraverserTraits<std::tuple_element_t<0, Childrens>>::dim;
    };

    template <SetTraverser_concept... SetTraversers>
    class DifferenceTraverser : public SetTraverserBase<DifferenceTraverser<SetTraversers...>>
    {
        using Self               = DifferenceTraverser<SetTraversers...>;
        using interval_t         = typename SetTraverserTraits<Self>::interval_t;
        using current_interval_t = typename SetTraverserTraits<Self>::current_interval_t;
        using Childrens          = typename SetTraverserTraits<Self>::Childrens;
        using value_t            = typename interval_t::value_t;

        template <size_t I>
        using IthChild = std::tuple_element<I, Childrens>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

      public:

        DifferenceTraverser(const std::array<std::size_t, nIntervals>& shifts, const SetTraversers&... set_traversers)
            : m_set_traversers(set_traversers...)
            , m_shifts(shifts)
        {
            compute_current_interval();

            enumerate_const_items(m_set_traversers,
                                  [](const auto i, const auto& set_traverser)
                                  {
                                      fmt::print("{} : {}th traverser_t = {}\n", __FUNCTION__, i, typeid(set_traverser).name());
                                  });
        }

        inline bool is_empty() const
        {
            return std::get<0>(m_set_traversers).is_empty();
        }

        inline void next_interval()
        {
            std::get<0>(m_set_traversers).next_interval();
            compute_current_interval();
        }

        inline current_interval_t current_interval() const
        {
            return m_current_interval;
        }

      private:

        inline void compute_current_interval()
        {
            while (!std::get<0>(m_set_traversers).is_empty() && !try_to_compute_current_interval())
            {
                std::get<0>(m_set_traversers).next_interval();
            }
        }

        inline bool try_to_compute_current_interval()
        {
            assert(!std::get<0>(m_set_traversers).is_empty());

            m_current_interval.start = std::get<0>(m_set_traversers).current_interval().start << m_shifts[0];
            m_current_interval.end   = std::get<0>(m_set_traversers).current_interval().end << m_shifts[0];

            fmt::print("initial current_interval = {}\n", m_current_interval);

            static_for<1, nIntervals>::apply(
                [this](const auto i)
                {
                    IthChild<i>& set_traverser = std::get<i>(m_set_traversers);

                    while (!set_traverser.is_empty() && (set_traverser.current_interval().end << m_shifts[i]) < m_current_interval.start)
                    {
                        set_traverser.next_interval();
                    }

                    if (!set_traverser.is_empty())
                    {
                        fmt::print("computing difference between {} and {}th interval {}\n",
                                   m_current_interval,
                                   i(),
                                   set_traverser.current_interval() << m_shifts[i]);
                        if ((set_traverser.current_interval().start << m_shifts[i]) <= m_current_interval.start)
                        {
                            assert((set_traverser.current_interval().end << m_shifts[i]) >= m_current_interval.start);
                            m_current_interval.start = set_traverser.current_interval().end << m_shifts[i];
                        }
                        else if ((set_traverser.current_interval().start << m_shifts[i]) <= m_current_interval.end)
                        {
                            assert(set_traverser.current_interval().start << m_shifts[i] <= m_current_interval.end);
                            m_current_interval.end = set_traverser.current_interval().start << m_shifts[i];
                        }
                        fmt::print("new current_interval = {}\n", m_current_interval);
                    }
                });

            return m_current_interval.is_valid();
        }

        interval_t m_current_interval;
        Childrens m_set_traversers;
        const std::array<std::size_t, nIntervals>& m_shifts;
    };

}
