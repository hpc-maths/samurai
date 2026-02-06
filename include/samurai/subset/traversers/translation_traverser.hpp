// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <class SetTraverser>
    class TranslationTraverser;

    template <class SetTraverser>
    struct SetTraverserTraits<TranslationTraverser<SetTraverser>>
    {
        static_assert(IsSetTraverser<SetTraverser>::value);

        using interval_t         = typename SetTraverserTraits<SetTraverser>::interval_t;
        using current_interval_t = interval_t;
    };

    template <class SetTraverser>
    class TranslationTraverser : public SetTraverserBase<TranslationTraverser<SetTraverser>>
    {
        using Self = TranslationTraverser<SetTraverser>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        TranslationTraverser(const SetTraverser& set_traverser, const value_t& translation)
            : m_set_traverser(set_traverser)
            , m_translation(translation)
        {
        }

        SAMURAI_INLINE bool is_empty_impl() const
        {
            return m_set_traverser.is_empty();
        }

        SAMURAI_INLINE void next_interval_impl()
        {
            m_set_traverser.next_interval();
        }

        SAMURAI_INLINE current_interval_t current_interval_impl() const
        {
            return current_interval_t{m_set_traverser.current_interval().start + m_translation,
                                      m_set_traverser.current_interval().end + m_translation};
        }

      private:

        SetTraverser m_set_traverser;
        value_t m_translation;
    };

} // namespace samurai
