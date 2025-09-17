// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../samurai_config.hpp"
#include "set_base.hpp"
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
        using traverser_t = std::conditional_t<d == Set::dim-1, 
                                               ProjectionTraverser<typename Set::template traverser_t<d>, 1>,
                                               ProjectionTraverser<typename Set::template traverser_t<d>, default_config::max_level>>; 
        
        static constexpr std::size_t dim = Set::dim;
    };

    template <class Set>
    class Projection : public SetBase<Projection<Set>>
    {
        using Self = Projection<Set>;
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
			if (m_projectionType == ProjectionType::COARSEN)
            {
                if constexpr (d != dim - 1)
                {
                    const value_t ymin = _index[d] << m_shift;
                    const value_t ymax = (_index[d] + 1) << m_shift;
		
                    xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> index(_index << m_shift);
		
                    //~ std::vector<typename Set::template traverser_t<d>> set_traversers;
                    //~ set_traversers.reserve(size_t(ymax - ymin));
                    FixedCapacityArray<typename Set::template traverser_t<d>, default_config::max_level> set_traversers;
		
                    for (index[d] = ymin; index[d] != ymax; ++index[d])
                    {
                        set_traversers.push_back(m_set.get_traverser(index, d_ic));
                    }
                    return traverser_t<d>(set_traversers, m_shift);
                }
                else
                {
                    return traverser_t<d>(m_set.get_traverser(_index << m_shift, d_ic), m_projectionType, m_shift);
                }
            }
            else
            {
                return traverser_t<d>(m_set.get_traverser(_index >> m_shift, d_ic), m_projectionType, m_shift);
            }
        }
        
	private:

        Set m_set;
        std::size_t m_level;
        ProjectionType m_projectionType;
        std::size_t m_shift;
	};

} // namespace samurai
