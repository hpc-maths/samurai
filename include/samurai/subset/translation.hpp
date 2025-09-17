// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/translation_traverser.hpp"

#include <xtensor/xview.hpp> 
using namespace xt::placeholders;  // this makes `_` available

namespace samurai
{

	template <class Set>
    class Translation;

    template <class Set>
    struct SetTraits<Translation<Set>>
    {
		static_assert(IsSet<Set>::value);
		
        template <std::size_t d>
        using traverser_t = TranslationTraverser<typename Set::template traverser_t<d>>;
        
        static constexpr std::size_t dim = Set::dim;
    };

    template <class Set>
    class Translation : public SetBase<Translation<Set>>
    {
        using Self = Translation<Set>;
    public:
		SAMURAI_SET_TYPEDEFS
		SAMURAI_SET_CONSTEXPRS
		
		using translation_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;
		
		template <class translation_expr_t>
        Translation(const Set& set, const translation_expr_t& translation_expr)
            : m_set(set)
            , m_translation(translation_expr)
        {
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
			return traverser_t<d>(m_set.get_traverser_impl(index - xt::view(m_translation, xt::range(1, _)), d_ic), m_translation[d]);
        }
        
    private:

        Set m_set;
        translation_t m_translation;
    };

    template <class Set, class translation_t>
    auto translate(const Set& set, const translation_t& translation)
    {
        return Translation(self(set), translation);
    }


} // namespace samurai
