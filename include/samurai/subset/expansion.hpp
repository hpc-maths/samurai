// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/expansion_traverser.hpp"
#include "traversers/last_dim_expansion_traverser.hpp"

namespace samurai
{
	
	template <class Set>
    class Expansion;

    template <class Set>
    struct SetTraits<Expansion<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using traverser_t = std::conditional_t<d == Set::dim-1
			, LastDimExpansionTraverser<typename Set::template traverser_t<d>>
			, ExpansionTraverser<typename Set::template traverser_t<d>>>;

        static constexpr std::size_t dim = Set::dim;
    };
    	
    namespace detail
    {
		template<class Set, typename Seq>
		struct ExpansionChildrenTraversers;
		
		template<class Set, std::size_t... ds>
		struct ExpansionChildrenTraversers<Set, std::index_sequence<ds...>>
		{
			template<std::size_t d>
			using child_traverser_array_t = std::vector< typename Set::template traverser_t<d> >;
				
			using Type = std::tuple< child_traverser_array_t<ds>... >;
			
			static_assert(std::tuple_size<Type>::value == Set::dim-1);
		};
		
	} // namespace detail
    	
	template <class Set>
    class Expansion : public SetBase<Expansion<Set>>
    {
		using Self = Expansion<Set>;
		using ChildrenTraversers = detail::ExpansionChildrenTraversers< Set, std::make_index_sequence<Set::dim-1> >::Type;
      public:

        SAMURAI_SET_TYPEDEFS
        SAMURAI_SET_CONSTEXPRS
        
		using expansion_t    = std::array<value_t, dim>;
        using do_expansion_t = std::array<bool, dim>;
        
        Expansion(const Set& set, const expansion_t& expansions)
			: m_set(set)
			, m_expansions(expansions)
		{
			static_for<0, dim-1>::apply([this](const auto d)
			{
				std::get<d>(m_children_traversers).reserve(2*m_expansions[d]);
			});
		}
        
        Expansion(const Set& set, const value_t expansion)
			: m_set(set)
		{
			m_expansions.fill(expansion);
			static_for<0, dim-1>::apply([this](const auto d)
			{
				std::get<d>(m_children_traversers).reserve(std::size_t(2*m_expansions[d]));
			});
		}
        
        Expansion(const Set& set, const value_t expansion, const do_expansion_t& do_expansion)
			: m_set(set)
		{
			for (std::size_t i = 0; i != m_expansions.size(); ++i)
            {
                m_expansions[i] = expansion * do_expansion[i];
            }
			
			static_for<0, dim-1>::apply([this](const auto d)
			{
				std::get<d>(m_children_traversers).reserve(2*m_expansions);
			});
		}
		
		inline std::size_t level_impl() const
        {
            return m_set.level();
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
        inline traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
			if constexpr (d == dim-1)
			{
				return traverser_t<d>(m_set.get_traverser(index, d_ic), m_expansions[d]);
			}
			else
			{
				auto& children_traversers = std::get<d>(m_children_traversers);
				
				children_traversers.clear();
				
				xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> tmp_index(index);
				
				for (value_t width=0; width!=m_expansions[d+1]+1; ++width)
				{
					tmp_index[d+1] = index[d+1] + width;
					children_traversers.push_back(m_set.get_traverser(tmp_index, d_ic));
					if (children_traversers.back().is_empty()) { children_traversers.pop_back(); }
					
					tmp_index[d+1] = index[d+1] - width;
					children_traversers.push_back(m_set.get_traverser(tmp_index, d_ic));
					if (children_traversers.back().is_empty()) { children_traversers.pop_back(); }
				}
				
				return traverser_t<d>(children_traversers.begin(), children_traversers.end(), m_expansions[d]);
			}
		}
			
	private:
		Set                m_set;
		expansion_t        m_expansions;
		
		mutable ChildrenTraversers m_children_traversers;
	}; 
	
	template<class Set>
	auto expand(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::contraction_t& expansions)
	{
		return Expansion(self(set), expansions);
	}
	
	template<class Set>
	auto expand(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::value_t expansion)
	{
		return Expansion(self(set), expansion);
	}
	
	template<class Set>
	auto expand(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::value_t expansion, const typename Contraction<std::decay_t<decltype(self(set))>>::do_expansion_t& do_expansion)
	{
		return Expansion(self(set), expansion, do_expansion);
	}
	
} // namespace samurai
