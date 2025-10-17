// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/translation_traverser.hpp"

#include <xtensor/xview.hpp>
using namespace xt::placeholders; // this makes `_` available

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

        struct Workspace
        {
            typename Set::Workspace child_workspace;
        };

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    template <class Set>
    class Translation : public SetBase<Translation<Set>>
    {
        using Self = Translation<Set>;

      public:

        SAMURAI_SET_TYPEDEFS

        using translation_t = xt::xtensor_fixed<value_t, xt::xshape<Base::dim>>;

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

        template <std::size_t d>
        inline void
        init_workspace_impl(const std::size_t n_traversers, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            m_set.init_workspace(n_traversers, d_ic, workspace.child_workspace);
        }

        template <std::size_t d>
        inline traverser_t<d>
        get_traverser_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            yz_index_t child_index;

            for (std::size_t i = 0; i != Base::dim - 1; ++i)
            {
                child_index[i] = index[i] - m_translation[i + 1];
            }

            return traverser_t<d>(m_set.get_traverser_impl(child_index, d_ic, workspace.child_workspace), m_translation[d]);
        }

        template <std::size_t d>
        inline traverser_t<d>
        get_traverser_unordered_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            yz_index_t child_index;

            for (std::size_t i = 0; i != Base::dim - 1; ++i)
            {
                child_index[i] = index[i] - m_translation[i + 1];
            }

            return traverser_t<d>(m_set.get_traverser_unordered_impl(child_index, d_ic, workspace.child_workspace), m_translation[d]);
        }

      private:

        Set m_set;
        translation_t m_translation;
    };

    template <class Set>
    auto translate(const Set& set, const typename Translation<std::decay_t<decltype(self(set))>>::translation_t& translation)
    {
        return Translation(self(set), translation);
    }

} // namespace samurai
