// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../fixed_capacity_array.hpp"
#include "../memory_pool.hpp"
#include "set_traverser_base.hpp"

namespace samurai
{
    enum class ProjectionType
    {
        COARSEN,
        REFINE
    };

    template <class SetTraverser>
    class ProjectionTraverser;

    template <class SetTraverser>
    struct SetTraverserTraits<ProjectionTraverser<SetTraverser>>
    {
        static_assert(IsSetTraverser<SetTraverser>::value);

        using interval_t         = typename SetTraverser::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class SetTraverser>
    class ProjectionTraverser : public SetTraverserBase<ProjectionTraverser<SetTraverser>>
    {
        using Self = ProjectionTraverser<SetTraverser>;

		using SetTraverserOffsetRange = MemoryPool<SetTraverser>::OffsetRange;
		using SetTraverserOffset      = MemoryPool<SetTraverser>::Distance;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        ProjectionTraverser(SetTraverserOffset set_traverser_offset, const ProjectionType projectionType, const std::size_t shift)
            : m_set_traverser_offsets(set_traverser_offset, set_traverser_offset + 1)
            , m_projectionType(projectionType)
            , m_shift(shift)
            , m_isEmpty(MemoryPool<SetTraverser>::getInstance().at(set_traverser_offset).is_empty())
        {			
			auto& memory_pool = MemoryPool<SetTraverser>::getInstance();
			
            if (!m_isEmpty)
            {				
				SetTraverser& set_traverser = memory_pool.at(set_traverser_offset);
				
                if (m_projectionType == ProjectionType::COARSEN)
                {
                    m_current_interval.start = coarsen_start(set_traverser.current_interval());
                    m_current_interval.end   = coarsen_end(set_traverser.current_interval());

                    set_traverser.next_interval();

                    // when coarsening, two disjoint intervals may be merged.
                    // we need to check if the next_interval overlaps
                    for (; !set_traverser.is_empty() && coarsen_start(set_traverser.current_interval()) <= m_current_interval.end;
                         set_traverser.next_interval())
                    {
                        m_current_interval.end = coarsen_end(set_traverser.current_interval());
                    }
                }
                else
                {					
                    m_current_interval.start = set_traverser.current_interval().start << shift;
                    m_current_interval.end   = set_traverser.current_interval().end << shift;
                }
            }
        }

        /*
         * This constructor only works for coarsening
         */
        ProjectionTraverser(const SetTraverserOffset first_set_traverser_offset, const SetTraverserOffset bound_set_traverser_offset, const std::size_t shift)
            : m_set_traverser_offsets(first_set_traverser_offset, bound_set_traverser_offset)
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
				SetTraverser& set_traverser = MemoryPool<SetTraverser>::getInstance().at(m_set_traverser_offsets.first);
				
                set_traverser.next_interval();
                m_isEmpty = set_traverser.is_empty();
                if (!m_isEmpty)
                {
                    m_current_interval.start = set_traverser.current_interval().start << m_shift;
                    m_current_interval.end   = set_traverser.current_interval().end << m_shift;
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
			auto& memory_pool = MemoryPool<SetTraverser>::getInstance();
			
            m_current_interval.start = std::numeric_limits<value_t>::max();
            // We find the start of the interval, i.e. the smallest set_traverser.current_interval().start >> m_shift
            for (auto offset = m_set_traverser_offsets.first; offset != m_set_traverser_offsets.bound; ++offset)
            {
				const SetTraverser& set_traverser = memory_pool.at(offset);
				
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
                for (auto offset = m_set_traverser_offsets.first; offset != m_set_traverser_offsets.bound; ++offset)
                {
					SetTraverser& set_traverser = memory_pool.at(offset);
					
                    while (!set_traverser.is_empty() && (coarsen_end(set_traverser.current_interval()) <= m_current_interval.end))
                    {
                        set_traverser.next_interval();
                    }
                }
                // try to find a new end
                for (auto offset = m_set_traverser_offsets.first; offset != m_set_traverser_offsets.bound; ++offset)
                {
					const SetTraverser& set_traverser = memory_pool.at(offset);
					
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
            return ((interval.end - 1) >> m_shift) + 1;
        }

        SetTraverserOffsetRange   m_set_traverser_offsets;
        ProjectionType            m_projectionType;
        std::size_t               m_shift;
        interval_t                m_current_interval;
        bool                      m_isEmpty;
    };

} // namespace samurai
