// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../utils.hpp"
#include "set_traverser_base.hpp"

namespace samurai
{

    template <SetTraverser_concept SetTraverser>
    class ContractionTraverser;

    template <SetTraverser_concept SetTraverser>
    struct SetTraverserTraits<ContractionTraverser<SetTraverser>>
    {
        using interval_t         = typename SetTraverserTraits<SetTraverser>::interval_t;
        using current_interval_t = interval_t;
    };

    template <SetTraverser_concept SetTraverser>
    class ContractionTraverser : public SetTraverserBase<ContractionTraverser<SetTraverser>>
    {
        using Self               = ContractionTraverser<SetTraverser>;
        using Base               = SetTraverserBase<Self>;
        using interval_t         = typename Base::interval_t;
        using current_interval_t = typename Base::current_interval_t;
        using value_t            = typename Base::value_t;

      public:

        ContractionTraverser(const SetTraverser& set, const std::size_t contraction)
            : m_set_traverser(m_set_traverser)
            , m_contraction(contraction)
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
            while (!m_set_traverser.is_empty() && m_set_traverser.current_interval().size() <= 2 * m_contraction)
            {
                m_set_traverser.next_interval();
            }
        }

        inline current_interval_t current_interval() const
        {
            return current_interval_t(m_set_traverser.current_interval().start + m_contraction,
                                      m_set_traverser.current_interval().end - m_contraction);
        }

      private:

        SetTraverser m_set_traverser;
        std::size_t m_contraction;
    };

}
