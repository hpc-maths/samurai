// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../samurai_config.hpp"
#include "../static_algorithm.hpp"
#include "set_base.hpp"
#include "memory_pool.hpp"
#include "traversers/projection_traverser.hpp"

namespace samurai
{

    template <class Set>
    class Projection;

    template <class Set>
    struct SetTraits<Projection<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using traverser_t = ProjectionTraverser< typename Set::template traverser_t<d> >;

        static constexpr std::size_t dim = Set::dim;
    };
    
    namespace detail
    {
		template<class Set, typename Seq>
		struct ProjectionWork;
		
		template<class Set, std::size_t... ds>
		struct ProjectionWork<Set, std::index_sequence<ds...>>
		{
			template<std::size_t d>
			using child_traverser_t = typename Set::template traverser_t<d>;
			
			template<std::size_t d>
			using array_of_child_traverser_offset_range_t = std::vector< typename MemoryPool< child_traverser_t<d> >::OffsetRange >;
			
			using Type = std::tuple< array_of_child_traverser_offset_range_t<ds>... >;
		};
	} // namespace detail

    template <class Set>
    class Projection : public SetBase<Projection<Set>>
    {
        using Self = Projection<Set>;
        using OffsetRangeWork =  detail::ProjectionWork< Set, std::make_index_sequence<Set::dim> >::Type;
      public:

        SAMURAI_SET_TYPEDEFS
        SAMURAI_SET_CONSTEXPRS

        Projection(const Set& set, const std::size_t level)
            : m_set(set)
            , m_level(level)
        {
            if (m_level < m_set.level())
            {
                m_projectionType = ProjectionType::COARSEN;
                m_shift          = m_set.level() - m_level;
            }
            else
            {
                m_projectionType = ProjectionType::REFINE;
                m_shift          = m_level - m_set.level();
            }
        }

		// we need to define a custom copy and move constructor because 
		// we do not want to copy m_work_offsetRanges
		Projection(const Projection& other)
			: m_set(other.m_set)
			, m_level(other.m_level)
			, m_projectionType(other.m_projectionType)
			, m_shift(other.m_shift)
		{
		}
		
		Projection(Projection&& other)
			: m_set(std::move(other.m_set))
			, m_level(std::move(other.m_level))
			, m_projectionType(std::move(other.m_projectionType))
			, m_shift(std::move(other.m_shift))
		{
		}

		~Projection()
		{
			static_for<0, dim>::apply([this](const auto d)
			{
				using Work = MemoryPool< typename Set::template traverser_t< d > >;
				
				auto& work = Work::getInstance();
				
				for (auto& offset_range : std::get<d>(m_work_offsetRanges))
				{
					work.freeChunk(offset_range);
				}
			}); 
		}

        inline std::size_t level_impl() const
        {
            return m_level;
        }

        inline bool exist_impl() const
        {
            return m_set.exist();
        }

        inline bool empty_impl() const
        {
            return m_set.empty();
        }

        template <class index_t, std::size_t d>
        inline traverser_t<d> get_traverser_impl(const index_t& _index, std::integral_constant<std::size_t, d> d_ic) const
        {
			using Work = MemoryPool< typename Set::template traverser_t<d> >;
			
			Work& work = Work::getInstance();
			
			auto& offsetRange = std::get<d>(m_work_offsetRanges);
			
            if (m_projectionType == ProjectionType::COARSEN)
            {
                if constexpr (d != dim - 1)
                {
					const auto set_traversers_offsets = work.requestChunk( 1 << m_shift );
					auto end_offset = set_traversers_offsets.first;
					
                    const value_t ymin = _index[d] << m_shift;
                    const value_t ymax = (_index[d] + 1) << m_shift;

                    xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> index(_index << m_shift);

                    for (index[d] = ymin; index[d] != ymax; ++index[d])
                    {
						std::construct_at(work.getPtr(end_offset), m_set.get_traverser(index, d_ic));
						if (work.at(end_offset).is_empty()) { std::destroy_at(work.getPtr(end_offset)); }
						else                                { ++end_offset;                             }
                    }
                    
                    offsetRange.push_back(set_traversers_offsets);
                    
                    return traverser_t<d>(set_traversers_offsets.first, end_offset, m_shift);
                }
                else
                {
					const auto set_traversers_offsets = work.requestChunk( 1 );
					std::construct_at(work.getPtr(set_traversers_offsets.first), m_set.get_traverser(_index << m_shift, d_ic));
					
					offsetRange.push_back(set_traversers_offsets);
					
                    return traverser_t<d>(set_traversers_offsets.first, m_projectionType, m_shift);
                }
            }
            else
            {
				const auto set_traversers_offsets = work.requestChunk( 1 );
				std::construct_at(work.getPtr(set_traversers_offsets.first), m_set.get_traverser(_index >> m_shift, d_ic));
				
				offsetRange.push_back(set_traversers_offsets);
					
                return traverser_t<d>(set_traversers_offsets.first, m_projectionType, m_shift);
            }
        }

      private:

        Set m_set;
        std::size_t m_level;
        ProjectionType m_projectionType;
        std::size_t m_shift;
        
        mutable OffsetRangeWork m_work_offsetRanges;
    };

} // namespace samurai
