// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

#include "utils.hpp"

namespace samurai
{
    template <class SetTraverser>
    class LastDimProjectionLOITraverser;

    template <class SetTraverser>
    struct SetTraverserTraits<LastDimProjectionLOITraverser<SetTraverser>>
    {
        static_assert(IsSetTraverser<SetTraverser>::value);

        using interval_t         = typename SetTraverser::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class SetTraverser>
    class LastDimProjectionLOITraverser : public SetTraverserBase<LastDimProjectionLOITraverser<SetTraverser>>
    {
        using Self = LastDimProjectionLOITraverser<SetTraverser>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        LastDimProjectionLOITraverser(const SetTraverser& set_traverser, const ProjectionType projectionType, const std::size_t shift)
            : m_set_traverser(set_traverser)
            , m_projectionType(projectionType)
            , m_shift(shift)
            , m_isEmpty(set_traverser.is_empty())
        {
            if (m_projectionType == ProjectionType::COARSEN)
            {
                next_interval_coarsen();
            }
            else if (!m_isEmpty)
            {
                m_current_interval.start = traverser_utils::refine_start(m_set_traverser.current_interval().start, shift);
                m_current_interval.end   = traverser_utils::refine_end(m_set_traverser.current_interval().end, shift);
            }
        }

        inline bool is_empty_impl() const
        {
            return m_isEmpty;
        }

        inline void next_interval_impl()
        {
            if (m_projectionType == ProjectionType::COARSEN)
            {
                next_interval_coarsen();
            }
            else
            {
                m_set_traverser.next_interval();
                m_isEmpty = m_set_traverser.is_empty();
                if (!m_isEmpty)
                {
                    m_current_interval.start = traverser_utils::refine_start(m_set_traverser.current_interval().start, m_shift);
                    m_current_interval.end   = traverser_utils::refine_end(m_set_traverser.current_interval().end, m_shift);
                }
            }
        }

        inline current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        inline void next_interval_coarsen()
        {
            if (!m_set_traverser.is_empty())
            {
                m_current_interval.start = traverser_utils::coarsen_start(m_set_traverser.current_interval().start, m_shift);
                m_current_interval.end   = traverser_utils::coarsen_end(m_set_traverser.current_interval().end, m_shift);

                m_set_traverser.next_interval();

                // when coarsening, two disjoint intervals may be merged.
                // we need to check if the next_interval overlaps
                for (; !m_set_traverser.is_empty()
                       && traverser_utils::coarsen_start(m_set_traverser.current_interval().start, m_shift) <= m_current_interval.end;
                     m_set_traverser.next_interval())
                {
                    m_current_interval.end = traverser_utils::coarsen_end(m_set_traverser.current_interval().end, m_shift);
                }
            }
            else
            {
                m_isEmpty = true;
            }
        }

        SetTraverser m_set_traverser;
        ProjectionType m_projectionType;
        std::size_t m_shift;
        interval_t m_current_interval;
        bool m_isEmpty;
    };

} // namespace samurai
