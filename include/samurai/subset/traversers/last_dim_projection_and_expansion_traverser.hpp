// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

#include "utils.hpp"

namespace samurai
{

    template <class SetTraverser>
    class LastDimProjectionAndExpansionTraverser;

    template <class SetTraverser>
    struct SetTraverserTraits<LastDimProjectionAndExpansionTraverser<SetTraverser>>
    {
        static_assert(IsSetTraverser<SetTraverser>::value);

        using interval_t         = typename SetTraverser::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class SetTraverser>
    class LastDimProjectionAndExpansionTraverser : public SetTraverserBase<LastDimProjectionAndExpansionTraverser<SetTraverser>>
    {
        using Self = LastDimProjectionAndExpansionTraverser<SetTraverser>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        LastDimProjectionAndExpansionTraverser(const SetTraverser& set_traverser,
                                               const ProjectionType projectionType,
                                               const std::size_t shift,
                                               const value_t expansion)
            : m_set_traverser(set_traverser)
            , m_projectionType(projectionType)
            , m_shift(shift)
            , m_expansion(expansion)
            , m_isEmpty(set_traverser.is_empty())
        {
            next_interval_impl();
        }

        SAMURAI_INLINE bool is_empty_impl() const
        {
            return m_isEmpty;
        }

        SAMURAI_INLINE void next_interval_impl()
        {
            if (m_projectionType == ProjectionType::COARSEN)
            {
                const auto startFunc = [shift = m_shift, expansion = m_expansion](const value_t& start) -> value_t
                {
                    return traverser_utils::coarsen_start(start, shift) - expansion;
                };
                const auto endFunc = [shift = m_shift, expansion = m_expansion](const value_t& end) -> value_t
                {
                    return traverser_utils::coarsen_end(end, shift) + expansion;
                };

                next_interval_impl_detail(startFunc, endFunc);
            }
            else
            {
                const auto startFunc = [shift = m_shift, expansion = m_expansion](const value_t& start) -> value_t
                {
                    return traverser_utils::refine_start(start, shift) - expansion;
                };
                const auto endFunc = [shift = m_shift, expansion = m_expansion](const value_t& end) -> value_t
                {
                    return traverser_utils::refine_end(end, shift) + expansion;
                };

                next_interval_impl_detail(startFunc, endFunc);
            }
        }

        SAMURAI_INLINE current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        template <std::invocable<value_t> StartFunc, std::invocable<value_t> EndFunc>
        void next_interval_impl_detail(const StartFunc startFunc, const EndFunc endFunc)
        {
            m_isEmpty = m_set_traverser.is_empty();

            if (!m_isEmpty)
            {
                m_current_interval.start = startFunc(m_set_traverser.current_interval().start);
                m_current_interval.end   = endFunc(m_set_traverser.current_interval().end);

                m_set_traverser.next_interval();
                while (!m_set_traverser.is_empty() and startFunc(m_set_traverser.current_interval().start) <= m_current_interval.end)
                {
                    m_current_interval.end = endFunc(m_set_traverser.current_interval().end);
                    m_set_traverser.next_interval();
                }
            }
        }

        SetTraverser m_set_traverser;
        ProjectionType m_projectionType;
        std::size_t m_shift;
        value_t m_expansion;
        interval_t m_current_interval;
        bool m_isEmpty;
    };

} // namespace samurai
