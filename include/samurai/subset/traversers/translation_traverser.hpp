// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include "set_traverser_base.hpp"

#pragma once

namespace samurai
{

    template <SetTraverser_concept SetTraverser>
    class TranslationTraverser;

    template <SetTraverser_concept SetTraverser>
    struct SetTraverserTraits<TranslationTraverser<SetTraverser>>
    {
        using interval_t         = typename SetTraverserTraits<SetTraverser>::interval_t;
        using current_interval_t = interval_t;
    };

    template <SetTraverser_concept SetTraverser>
    class TranslationTraverser : public SetTraverserBase<TranslationTraverser<SetTraverser>>
    {
        using Self = TranslationTraverser<SetTraverser>;
        using Base = SetTraverserBase<Self>;

      public:

        using interval_t         = typename Base::interval_t;
        using current_interval_t = typename Base::current_interval_t;
        using value_t            = typename Base::value_t;

        TranslationTraverser(const SetTraverser& set_traverser, const value_t& translation)
            : m_set_traverser(set_traverser)
            , m_translation(translation)
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
            return current_interval_t{m_set_traverser.current_interval().start + m_translation,
                                      m_set_traverser.current_interval().end + m_translation};
        }

      private:

        SetTraverser m_set_traverser;
        value_t m_translation;
    };
}
