// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "lca_traverser.hpp"
#include "set_base.hpp"

namespace samurai
{
    template <Box_concept B>
    class BoxView;

    template <Box_concept B>
    struct SetTraits<BoxView<B>>
    {
        using traverser_t = BoxTraverser<B>;
    };

    template <Box_concept B>
    class BoxView : public SetBase<BoxView<B>>
    {
        using Base = SetBase<BoxView<B>>;

      public:

        using index_t     = typename Base::index_t;
        using traverser_t = typename Base::traverser_t;

        BoxView(const std::size_t level, const B& box)
            : m_level(level)
            , m_box(box)
        {
        }

        std::size_t level() const
        {
            return m_level;
        }

        bool exists() const
        {
            return m_box.is_valid();
        }

        bool empty() const
        {
            return !exists();
        }

        template <std::size_t d>
        traverser_t get_traverser(const index_t& index, std::integral_constant<std::size_t d>) const
        {
            return traverser_t(m_box.min_corner()[d], m_box.max_corner()[d]);
        }

      private:

        const B& m_box;
    };

    template <Box_concept B>
    BoxView<B> self(const B& box)
    {
        return BoxView<B>(box);
    }

}
