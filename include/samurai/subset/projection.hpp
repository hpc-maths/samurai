// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../samurai_config.hpp"
#include "../static_algorithm.hpp"
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
			using child_traverser_array_t = std::vector< typename Set::template traverser_t<d> >;
			
			using Type = std::tuple< child_traverser_array_t<ds>... >;
		};
	} // namespace detail

    template <class Set>
    class Projection : public SetBase<Projection<Set>>
    {
        using Self = Projection<Set>;
        using Work = typename detail::ProjectionWork< Set, std::make_index_sequence<Set::dim> >::Type;
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
			auto& set_traversers = std::get<d>(m_work_traversers);
			set_traversers.clear();
			
            if (m_projectionType == ProjectionType::COARSEN)
            {
                if constexpr (d != dim - 1)
                {
					//~ set_traversers.reserve(1 << m_shift);
					
                    const value_t ymin = _index[d] << m_shift;
                    const value_t ymax = (_index[d] + 1) << m_shift;

                    xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> index(_index << m_shift);

                    for (index[d] = ymin; index[d] != ymax; ++index[d])
                    {
                        set_traversers.push_back(m_set.get_traverser(index, d_ic));
						if (set_traversers.back().is_empty()) { set_traversers.pop_back(); }
                    }
                    return traverser_t<d>(set_traversers.begin(), set_traversers.end(), m_shift);
                }
                else
                {
					set_traversers.push_back(m_set.get_traverser(_index << m_shift, d_ic));
                    return traverser_t<d>(set_traversers.begin(), m_projectionType, m_shift);
                }
            }
            else
            {
				set_traversers.push_back(m_set.get_traverser(_index >> m_shift, d_ic));
                return traverser_t<d>(set_traversers.begin(), m_projectionType, m_shift);
            }
        }

      private:

        Set m_set;
        std::size_t m_level;
        ProjectionType m_projectionType;
        std::size_t m_shift;
        
        Work m_work_traversers;
    };

} // namespace samurai
