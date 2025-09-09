// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/box_traverser.hpp"

namespace samurai
{
    template <Box_concept B>
    class BoxView;

    template <Box_concept B>
    struct SetTraits<BoxView<B>>
    {
        template <std::size_t>
        using traverser_t = BoxTraverser<B>;

        static constexpr std::size_t getDim()
        {
            return B::dim;
        }
    };

    template <Box_concept B>
    class BoxView : public SetBase<BoxView<B>>
    {
        using Self = BoxView<B>;
        using Base = SetBase<Self>;

      public:

        template <std::size_t d>
        using traverser_t = typename Base::template traverser_t<d>;

        BoxView(const std::size_t level, const B& box)
            : m_level(level)
            , m_box(box)
        {
        }

        std::size_t level() const
        {
            return m_level;
        }

        bool exist() const
        {
            return m_box.is_valid();
        }

        bool empty() const
        {
            return !exist();
        }

        template <class index_t, std::size_t d>
        traverser_t<d> get_traverser([[maybe_unused]] const index_t& index, std::integral_constant<std::size_t, d>) const
        {
            if constexpr (d != Base::dim - 1)
            {
                assert(m_box.min_corner()[d + 1] <= index[d] && index[d] < m_box.max_corner()[d + 1]);
            }

            return traverser_t<d>(m_box.min_corner()[d], m_box.max_corner()[d]);
        }

      private:

        std::size_t m_level;
        const B& m_box;
    };

    template <Box_concept B>
    BoxView<B> self(const B& box)
    {
        return BoxView<B>(box);
    }

}
