// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../list_of_intervals.hpp"
#include "set_traverser_base.hpp"

namespace samurai
{
    template <class SetTraverser>
    class ProjectionLOITraverser;

    template <class SetTraverser>
    struct SetTraverserTraits<ProjectionLOITraverser<SetTraverser>>
    {
        static_assert(IsSetTraverser<SetTraverser>::value);

        using interval_t         = typename SetTraverser::interval_t;
        using current_interval_t = interval_t;
    };

    template <class SetTraverser>
    class ProjectionLOITraverser : public SetTraverserBase<ProjectionLOITraverser<SetTraverser>>
    {
        using Self                 = ProjectionLOITraverser<SetTraverser>;
        using SetTraverserIterator = typename ListOfIntervals<typename SetTraverser::value_t>::const_iterator;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        ProjectionLOITraverser(SetTraverser set_traverser, const ProjectionType projectionType, const std::size_t shift)
            : m_set_traverser(set_traverser)
            , m_shift(shift)
            , m_projectionType(ProjectionType::REFINE)
        {
            assert(projectionType == m_projectionType);
        }

        ProjectionLOITraverser(SetTraverser set_traverser, SetTraverserIterator first_interval, SetTraverserIterator bound_interval)
            : m_set_traverser(set_traverser)
            , m_first_interval(first_interval)
            , m_bound_interval(bound_interval)
            , m_projectionType(ProjectionType::COARSEN)
        {
        }

        inline bool is_empty_impl() const
        {
            if (m_projectionType == ProjectionType::COARSEN)
            {
                return m_first_interval == m_bound_interval;
            }
            else
            {
                return m_set_traverser.is_empty();
            }
        }

        inline void next_interval_impl()
        {
            if (m_projectionType == ProjectionType::COARSEN)
            {
                ++m_first_interval;
            }
            else
            {
                m_set_traverser.next_interval();
            }
        }

        inline current_interval_t current_interval_impl() const
        {
            return (m_projectionType == ProjectionType::COARSEN) ? *m_first_interval : m_set_traverser.current_interval() << m_shift;
        }

      private:

        SetTraverser m_set_traverser;          // only used when refining
        std::size_t m_shift;                   // only used when refining
        SetTraverserIterator m_first_interval; // only use when coarsening
        SetTraverserIterator m_bound_interval; // only use when coarsening
        ProjectionType m_projectionType;
    };

} // namespace samurai
