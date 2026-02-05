// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <class SetTraverser>
    class ExpansionTraverser;

    template <class SetTraverser>
    struct SetTraverserTraits<ExpansionTraverser<SetTraverser>>
    {
        static_assert(IsSetTraverser<SetTraverser>::value);

        using interval_t         = typename SetTraverserTraits<SetTraverser>::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class SetTraverser>
    class ExpansionTraverser : public SetTraverserBase<ExpansionTraverser<SetTraverser>>
    {
        using Self = ExpansionTraverser<SetTraverser>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        using SetTraverserIterator = typename std::vector<SetTraverser>::iterator;

        ExpansionTraverser(SetTraverserIterator begin_set_traverser, SetTraverserIterator end_set_traverser, const value_t expansion)
            : m_set_traversers(begin_set_traverser, end_set_traverser)
            , m_expansion(expansion)
        {
            next_interval_impl();
        }

        SAMURAI_INLINE bool is_empty_impl() const
        {
            return m_current_interval.start == std::numeric_limits<value_t>::max();
        }

        SAMURAI_INLINE void next_interval_impl()
        {
            const auto startFunc = [expansion = m_expansion](const std::size_t /* i */, const value_t& start) -> value_t
            {
                return start - expansion;
            };
            const auto endFunc = [expansion = m_expansion](const std::size_t /* i */, const value_t& end) -> value_t
            {
                return end + expansion;
            };
            // maybe a small overhead due to the unused i and an enumerate instead of a range based for loop.
            // but we have the same algo for expansion_traverser and union_traverser.
            // Maybe the compiler can figure out that we do never use i and is able to simplify a but, but I wouldn't bet
            // an eye on it...
            // I don't know if the potential perf. compromise is worth it.
            m_current_interval = traverser_utils::transform_and_union(m_set_traversers, startFunc, endFunc);
        }

        SAMURAI_INLINE current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        std::span<SetTraverser> m_set_traversers;
        value_t m_expansion;
        interval_t m_current_interval;
    };

} // namespace samurai
