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
            assert(m_expansion > 0);
            next_interval_impl();
        }

        inline bool is_empty_impl() const
        {
            return m_current_interval.start == std::numeric_limits<value_t>::max();
        }

        inline void next_interval_impl()
        {
            m_current_interval.start = std::numeric_limits<value_t>::max();

            // We find the start of the interval, i.e. the smallest set_traverser.current_interval().start
            for (const SetTraverser& set_traverser : m_set_traversers)
            {
                if (!set_traverser.is_empty() && (set_traverser.current_interval().start - m_expansion < m_current_interval.start))
                {
                    m_current_interval.start = set_traverser.current_interval().start - m_expansion;
                    m_current_interval.end   = set_traverser.current_interval().end + m_expansion;
                }
            }
            // Now we find the end of the interval, i.e. the largest set_traverser.current_interval().end
            // such that set_traverser.current_interval().start - expansion < m_current_interval.end
            bool is_done = false;
            while (!is_done)
            {
                is_done = true;
                // advance set traverses that are behind current interval
                for (SetTraverser& set_traverser : m_set_traversers)
                {
                    while (!set_traverser.is_empty() && set_traverser.current_interval().end + m_expansion <= m_current_interval.end)
                    {
                        set_traverser.next_interval();
                    }
                }
                // try to find a new end
                for (const SetTraverser& set_traverser : m_set_traversers)
                {
                    // there is an overlap
                    if (!set_traverser.is_empty() && set_traverser.current_interval().start - m_expansion <= m_current_interval.end)
                    {
                        is_done                = false;
                        m_current_interval.end = set_traverser.current_interval().end + m_expansion;
                    }
                }
            }
        }

        inline current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        std::span<SetTraverser> m_set_traversers;
        value_t m_expansion;
        interval_t m_current_interval;
        bool m_isEmpty;
    };

} // namespace samurai
