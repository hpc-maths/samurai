// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>

#include <utility>
#include <type_traits>

#include "traversers/projection_traverser.hpp"

namespace samurai
{
    template <class Set>
    class Projection;

    template <class Set>
    struct SetTraits<Projection<Set>>
    {
        template <std::size_t d>
        using traverser_t = ProjectionTraverser<typename Set::template traverser_t<d>>;

        static constexpr std::size_t getDim()
        {
            return SetTraits<Set>::getDim();
        }
    };

	namespace detail
	{
		template<class Set, typename Seq> struct Work;
		
		template<class Set, std::size_t... ds>
		struct Work< Set, std::index_sequence<ds...> >
		{
			template <std::size_t d>
			using WorkElement = std::vector< typename Set::template traverser_t<d> >;
			
			using Type = std::tuple< WorkElement<ds>... >; 
		};
	}
	
	template<class Set>
	using Work = typename detail::Work< Set, std::make_index_sequence<SetTraits<Set>::getDim()> >::Type;

    /*
     * The main issue with projection, Coarsening specifically, is that
     * the interval is extended upon projection e.g. both
     * Proj([1,4),1,0) and Proj([0,3),1,0) results in [0, 2)
     * It means the coarsening operation, unlike the translation e.g.,
     * is a unary, NON BIJECTIVE operation.
     * Thus in more than 1d, if I'm projecting over more than 1 level_impl:
     * Proj([5, 17), [18, 20), 2, 0) = [1, 3)
     * So given a y_proj in [1, 3), we need to traverse accross multiple
     * y.
     * y_proj = 1 => y \in [4, 8)
     * y_proj = 2 => y \in [8, 12)
     *
     */
    template <class Set>
    class Projection : public SetBase<Projection<Set>>
    {
        using Self = Projection<Set>;
        using Base = SetBase<Self>;

      public:

        template <std::size_t d>
        using traverser_t = typename Base::template traverser_t<d>;

        using value_t = typename Base::value_t;

        Projection(const Set& set, const std::size_t level_impl)
            : m_set(set)
            , m_level_impl(level_impl)
        {
            if (m_level_impl < m_set.level_impl())
            {
                m_projectionType = ProjectionType::COARSEN;
                m_shift          = m_set.level_impl() - m_level_impl;
            }
            else
            {
                m_projectionType = ProjectionType::REFINE;
                m_shift          = m_level_impl - m_set.level_impl();
            }
        }

        std::size_t level_impl() const
        {
            return m_level_impl;
        }

        bool exist_impl() const
        {
            return m_set.exist_impl();
        }

        bool empty_impl() const
        {
            return m_set.empty_impl();
        }

        template <class index_t, std::size_t d>
        traverser_t<d> get_traverser_impl(const index_t& _index, std::integral_constant<std::size_t, d> d_ic) const
        {
            if (m_projectionType == ProjectionType::COARSEN)
            {
                if constexpr (d != Base::dim - 1)
                {
                    const value_t ymin = _index[d] << m_shift;
                    const value_t ymax = (_index[d] + 1) << m_shift;
		
                    xt::xtensor_fixed<value_t, xt::xshape<Base::dim - 1>> index(_index << m_shift);
		
                    //~ std::vector<typename SetTraits<Set>::template traverser_t<d>> set_traversers;
                    //~ set_traversers.reserve(size_t(ymax - ymin));
                    FixedCapacityArray<typename SetTraits<Set>::template traverser_t<d>, default_config::max_level> set_traversers;
		
                    for (index[d] = ymin; index[d] != ymax; ++index[d])
                    {
                        set_traversers.push_back(m_set.get_traverser_impl(index, d_ic));
                    }
                    return traverser_t<d>(set_traversers, m_shift);
                }
                else
                {
                    return traverser_t<d>(m_set.get_traverser_impl(_index << m_shift, d_ic), m_projectionType, m_shift);
                }
            }
            else
            {
                return traverser_t<d>(m_set.get_traverser_impl(_index >> m_shift, d_ic), m_projectionType, m_shift);
            }
        }

      private:

        Set m_set;
        std::size_t m_level_impl;
        ProjectionType m_projectionType;
        std::size_t m_shift;
    };
}
