// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <class SetTraverser>
    class ContractionTraverser;

    template <class SetTraverser>
    struct SetTraverserTraits<ContractionTraverser<SetTraverser>>
    {
        static_assert(IsSetTraverser<SetTraverser>::value);

        using interval_t         = typename SetTraverser::interval_t;
        using current_interval_t = interval_t;
    };

    template <class SetTraverser>
    class ContractionTraverser : public SetTraverserBase<ContractionTraverser<SetTraverser>>
    {
        using Self = ContractionTraverser<SetTraverser>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        ContractionTraverser(const SetTraverser& set_traverser, const value_t contraction)
            : m_set_traverser(set_traverser)
            , m_contraction(contraction)
        {
            assert(m_contraction >= 0);
        }

        inline bool is_empty_impl() const
        {
            return m_set_traverser.is_empty();
        }

        inline void next_interval_impl()
        {
            m_set_traverser.next_interval();
            while (!m_set_traverser.is_empty() && m_set_traverser.current_interval().size() <= size_t(2 * m_contraction))
            {
                m_set_traverser.next_interval();
            }
        }

        inline current_interval_t current_interval_impl() const
        {
            return current_interval_t(m_set_traverser.current_interval().start + m_contraction,
                                      m_set_traverser.current_interval().end - m_contraction);
        }

      private:

        SetTraverser m_set_traverser;
        value_t m_contraction;
    };

} // namespace samurai
