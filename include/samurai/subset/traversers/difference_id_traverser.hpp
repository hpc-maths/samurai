// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <class FirstSetTraverser, class... OtherSetTraversers>
    class DifferenceIdTraverser;

    template <class FirstSetTraverser, class... OtherSetTraversers>
    struct SetTraverserTraits<DifferenceIdTraverser<FirstSetTraverser, OtherSetTraversers...>>
    {
        static_assert(IsSetTraverser<FirstSetTraverser>::value);
        static_assert((IsSetTraverser<OtherSetTraversers>::value and ...));

        using interval_t         = typename FirstSetTraverser::interval_t;
        using current_interval_t = interval_t;
    };

    template <class FirstSetTraverser, class... OtherSetTraversers>
    class DifferenceIdTraverser : public SetTraverserBase<DifferenceIdTraverser<FirstSetTraverser, OtherSetTraversers...>>
    {
        using Self = DifferenceIdTraverser<FirstSetTraverser>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        static constexpr std::size_t nIntervals = 1 + sizeof...(OtherSetTraversers);

        DifferenceIdTraverser(const std::array<std::size_t, nIntervals>& shifts,
                              const FirstSetTraverser& set_traverser,
                              const OtherSetTraversers&...)
            : m_set_traverser(set_traverser)
            , m_shift(shifts[0])
        {
        }

        DifferenceIdTraverser() = delete;

        inline bool is_empty_impl() const
        {
            return m_set_traverser.is_empty();
        }

        inline void next_interval_impl()
        {
            m_set_traverser.next_interval();
        }

        inline current_interval_t current_interval_impl() const
        {
            return current_interval_t{m_set_traverser.current_interval().start << m_shift, m_set_traverser.current_interval().end << m_shift};
        }

      private:

        FirstSetTraverser m_set_traverser;
        std::size_t m_shift;
    };

} // namespace samurai
