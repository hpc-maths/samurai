// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"
#include "../fixed_capacity_array.hpp"

#include <fmt/core.h>

namespace samurai
{
	enum class ProjectionType
    {
        COARSEN,
        REFINE
    };

    template <class SetTraverser, std::size_t nTraversers>
    class ProjectionTraverser;

    template <class SetTraverser, std::size_t nTraversers>
    struct SetTraverserTraits<ProjectionTraverser<SetTraverser, nTraversers>>
    {
		static_assert(IsSetTraverser<SetTraverser>::value);
		
        using interval_t         = typename SetTraverser::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class SetTraverser, std::size_t nTraversers>
    class ProjectionTraverser : public SetTraverserBase<ProjectionTraverser<SetTraverser, nTraversers>>
    {
		using Self = ProjectionTraverser<SetTraverser, nTraversers>;
	public:
		SAMURAI_SET_TRAVERSER_TYPEDEFS
		
		ProjectionTraverser(const SetTraverser& set_traverser, const ProjectionType projectionType, const std::size_t shift)
            : m_projectionType(projectionType)
            , m_shift(shift)
            , m_isEmpty(set_traverser.is_empty())
        {
            m_set_traversers.push_back(set_traverser);
            if (!m_isEmpty)
            {
                if (m_projectionType == ProjectionType::COARSEN)
                {
                    m_current_interval.start = coarsen_start(m_set_traversers[0].current_interval());
                    m_current_interval.end   = coarsen_end(m_set_traversers[0].current_interval());

                    m_set_traversers[0].next_interval();

                    // when coarsening, two disjoint intervals may be merged.
                    // we need to check if the next_interval overlaps
                    for (; !m_set_traversers[0].is_empty() && coarsen_start(m_set_traversers[0].current_interval()) <= m_current_interval.end;
                         m_set_traversers[0].next_interval())
                    {
                        m_current_interval.end = coarsen_end(m_set_traversers[0].current_interval());
                    }
                }
                else
                {
                    m_current_interval.start = m_set_traversers[0].current_interval().start << shift;
                    m_current_interval.end   = m_set_traversers[0].current_interval().end << shift;
                }
            }
        }
        
        /*
         * This constructor only works for coarsening
         */
        ProjectionTraverser(const FixedCapacityArray<SetTraverser, nTraversers>& set_traversers, const std::size_t shift)
            : m_set_traversers(set_traversers)
            , m_projectionType(ProjectionType::COARSEN)
            , m_shift(shift)
        {
            next_interval_coarsen();
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
                m_set_traversers[0].next_interval();
                m_isEmpty = m_set_traversers[0].is_empty();
                if (!m_isEmpty)
                {
                    m_current_interval.start = m_set_traversers[0].current_interval().start << m_shift;
                    m_current_interval.end   = m_set_traversers[0].current_interval().end << m_shift;
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
            m_current_interval.start = std::numeric_limits<value_t>::max();
            // We find the start of the interval, i.e. the smallest set_traverser.current_interval().start >> m_shift
            for (const SetTraverser& set_traverser : m_set_traversers)
            {
                if (!set_traverser.is_empty() && (coarsen_start(set_traverser.current_interval()) < m_current_interval.start))
                {
                    m_current_interval.start = coarsen_start(set_traverser.current_interval());
                    m_current_interval.end   = coarsen_end(set_traverser.current_interval());
                }
            }
            // Now we find the end of the interval, i.e. the largest set_traverser.current_interval().end >> m_shift
            // such that (set_traverser.current_interval().start >> m_shift) < m_current_interval.end
            bool is_done = false;
            while (!is_done)
            {
                is_done = true;
                // advance set traverses that are behind current interval
                for (SetTraverser& set_traverser : m_set_traversers)
                {
                    while (!set_traverser.is_empty() && (coarsen_end(set_traverser.current_interval()) <= m_current_interval.end))
                    {
                        set_traverser.next_interval();
                    }
                }
                // try to find a new end
                for (const SetTraverser& set_traverser : m_set_traversers)
                {
                    // there is an overlap
                    if (!set_traverser.is_empty() && (coarsen_start(set_traverser.current_interval()) <= m_current_interval.end))
                    {
                        is_done                = false;
                        m_current_interval.end = coarsen_end(set_traverser.current_interval());
                    }
                }
            }
            m_isEmpty = (m_current_interval.start == std::numeric_limits<value_t>::max());
        }
        
        inline value_t coarsen_start(const interval_t& interval) const
        {
            return interval.start >> m_shift;
        }

        inline value_t coarsen_end(const interval_t& interval) const
        {
            const value_t trial_end = interval.end >> m_shift;
            return (trial_end << m_shift) < interval.end ? trial_end + 1 : trial_end;
        }
	
		FixedCapacityArray<SetTraverser, nTraversers> m_set_traversers;
        ProjectionType m_projectionType;
        std::size_t m_shift;
        interval_t m_current_interval;
        bool m_isEmpty;
	};
    
} // namespace samurai
