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
        //~ using SetTraverserIterator = typename std::vector<SetTraverser>::iterator;

		using SetTraverserOffsetRange = MemoryPool<SetTraverser>::OffsetRange;
		using SetTraverserOffset      = MemoryPool<SetTraverser>::Distance;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        ProjectionTraverser(SetTraverserOffset set_traverser_offset, const ProjectionType projectionType, const std::size_t shift)
            : m_set_traverser_offsets(set_traverser_offset, set_traverser_offset + 1)
            , m_is_work_freed(false)
            , m_projectionType(projectionType)
            , m_shift(shift)
            , m_isEmpty(MemoryPool<SetTraverser>::getInstance().at(set_traverser_offset).is_empty())
        {
			fmt::print("constructed with chunk {} of {}\n", fmt::join(m_set_traverser_offsets, ", "), typeid(SetTraverser).name());
			
			const auto set_traversers = get_set_traversers_view();
			
            if (!m_isEmpty)
            {				
                if (m_projectionType == ProjectionType::COARSEN)
                {
                    m_current_interval.start = coarsen_start(set_traversers[0].current_interval());
                    m_current_interval.end   = coarsen_end(set_traversers[0].current_interval());

                    set_traversers[0].next_interval();

                    // when coarsening, two disjoint intervals may be merged.
                    // we need to check if the next_interval overlaps
                    for (; !set_traversers[0].is_empty() && coarsen_start(set_traversers[0].current_interval()) <= m_current_interval.end;
                         set_traversers[0].next_interval())
                    {
                        m_current_interval.end = coarsen_end(set_traversers[0].current_interval());
                    }
                }
                else
                {					
                    m_current_interval.start = set_traversers[0].current_interval().start << shift;
                    m_current_interval.end   = set_traversers[0].current_interval().end << shift;
                }
            }
        }

        /*
         * This constructor only works for coarsening
         */
        ProjectionTraverser(const SetTraverserOffset first_set_traverser_offset, const SetTraverserOffset last_set_traverser_offset, const std::size_t shift)
            : m_set_traverser_offsets(first_set_traverser_offset, last_set_traverser_offset)
            , m_is_work_freed(false)
            , m_projectionType(ProjectionType::COARSEN)
            , m_shift(shift)
        {
			fmt::print("constructed with chunk {} of {}\n", fmt::join(m_set_traverser_offsets, ", "), typeid(SetTraverser).name());
			
            next_interval_coarsen();
        }

        inline bool is_empty_impl() const
        {			
            return m_isEmpty;
        }

        inline void next_interval_impl()
        {
			const auto set_traversers = get_set_traversers_view();
			
            if (m_projectionType == ProjectionType::COARSEN)
            {
                next_interval_coarsen();
            }
            else
            {
                set_traversers[0].next_interval();
                m_isEmpty = set_traversers[0].is_empty();
                if (!m_isEmpty)
                {
                    m_current_interval.start = set_traversers[0].current_interval().start << m_shift;
                    m_current_interval.end   = set_traversers[0].current_interval().end << m_shift;
                }
                else if (not m_is_work_freed)
				{
					MemoryPool<SetTraverser>::getInstance().freeChunk(m_set_traverser_offsets);
					m_is_work_freed = true;
				}
            }
        }

        inline current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:
      
		inline auto get_set_traversers_view()
		{	
			return m_set_traverser_offsets | std::views::transform([&pool = MemoryPool<SetTraverser>::getInstance()](const SetTraverserOffset& offset) -> SetTraverser&
			{
				return pool.at(offset);
			});
		}

        inline void next_interval_coarsen()
        {
			const auto set_traversers = get_set_traversers_view();
			
            m_current_interval.start = std::numeric_limits<value_t>::max();
            // We find the start of the interval, i.e. the smallest set_traverser.current_interval().start >> m_shift
            for (const SetTraverser& set_traverser : set_traversers)
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
                for (SetTraverser& set_traverser : set_traversers)
                {
                    while (!set_traverser.is_empty() && (coarsen_end(set_traverser.current_interval()) <= m_current_interval.end))
                    {
                        set_traverser.next_interval();
                    }
                }
                // try to find a new end
                for (const SetTraverser& set_traverser : set_traversers)
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
            
            if (m_isEmpty and not m_is_work_freed)
			{
				MemoryPool<SetTraverser>::getInstance().freeChunk(m_set_traverser_offsets);
				m_is_work_freed = true;
			}
        }

        inline value_t coarsen_start(const interval_t& interval) const
        {
            return interval.start >> m_shift;
        }

		inline value_t coarsen_end(const interval_t& interval) const
        {
            return ((interval.end - 1) >> m_shift) + 1;
        }

        SetTraverserOffsetRange m_set_traverser_offsets;
        bool                    m_is_work_freed;
        ProjectionType          m_projectionType;
        std::size_t             m_shift;
        interval_t              m_current_interval;
        bool                    m_isEmpty;
    };

} // namespace samurai
