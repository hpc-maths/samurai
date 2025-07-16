// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "box_view.hpp"
#include "lca_view.hpp"
#include "set_base.hpp"
#include "translation_traverser.hpp"

#include <fmt/ranges.h>

namespace samurai
{

    template <Set_concept Set>
    class Translation;

    template <Set_concept Set>
    struct SetTraits<Translation<Set>>
    {
        using traverser_t = TranslationTraverser<typename SetTraits<Set>::traverser_t>;
    };

    template <Set_concept Set>
    class Translation : public SetBase<Translation<Set>>
    {
        using Base = SetBase<Translation<Set>>;

      public:

        using traverser_t   = typename Base::traverser_t;
        using value_t       = typename Base::value_t;
        using translation_t = xt::xtensor_fixed<value_t, xt::xshape<SetTraverserTraits<traverser_t>::dim>>;

        template <class translation_expr_t>
        Translation(const Set& set, const translation_expr_t& translation_expr)
            : m_set(set)
            , m_translation(translation_expr)
        {
            fmt::print("{} : translation = {}\n", __FUNCTION__, m_translation);
        }

        std::size_t level() const
        {
            return m_set.level();
        }

        bool exist() const
        {
            return m_set.exist();
        }

        bool empty() const
        {
            return m_set.empty();
        }

        template <class index_t, std::size_t d>
        traverser_t get_traverser(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            //~ return traverser_t(m_set.get_traverser(index - m_translation, d_ic), m_translation[d]);
            fmt::print("{} : translation = {}, d = {}\n", __FUNCTION__, m_translation, d);
            return traverser_t(m_set.get_traverser(index - xt::view(m_translation, xt::range(1, _)), d_ic), m_translation[d]);
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

}
