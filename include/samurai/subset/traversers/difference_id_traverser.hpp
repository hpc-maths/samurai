// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../static_algorithm.hpp"
#include "../utils.hpp"
#include "set_traverser_base.hpp"

namespace samurai
{

    template <SetTraverser_concept FirstSetTraverser, SetTraverser_concept... OtherSetTraversers>
    class DifferenceIdTraverser;

    template <SetTraverser_concept FirstSetTraverser, SetTraverser_concept... OtherSetTraversers>
    struct SetTraverserTraits<DifferenceIdTraverser<FirstSetTraverser, OtherSetTraversers...>>
    {
        using interval_t         = typename SetTraverserTraits<FirstSetTraverser>::interval_t;
        using current_interval_t = interval_t;
    };

    template <SetTraverser_concept FirstSetTraverser, SetTraverser_concept... OtherSetTraversers>
    class DifferenceIdTraverser : public SetTraverserBase<DifferenceIdTraverser<FirstSetTraverser, OtherSetTraversers...>>
    {
        using Self = DifferenceIdTraverser<FirstSetTraverser, OtherSetTraversers...>;
        using Base = SetTraverserBase<Self>;

      public:

        using interval_t         = typename Base::interval_t;
        using current_interval_t = typename Base::current_interval_t;
        using value_t            = typename Base::value_t;

        static constexpr std::size_t nIntervals = 1 + sizeof...(OtherSetTraversers);

        DifferenceIdTraverser(const std::array<std::size_t, nIntervals>& shifts,
                              const FirstSetTraverser& set_traverser,
                              const OtherSetTraversers&...)
            : m_set_traverser(set_traverser)
            , m_shift(shifts[0])
        {
        }

        inline bool is_empty() const
        {
            return m_set_traverser.is_empty();
        }

        inline void next_interval()
        {
            assert(!is_empty());
            m_set_traverser.next_interval();
        }

        inline current_interval_t current_interval() const
        {
            return current_interval_t{m_set_traverser.current_interval().start << m_shift, m_set_traverser.current_interval().end << m_shift};
        }

      private:

        FirstSetTraverser m_set_traverser;
        std::size_t m_shift;
    };

}
