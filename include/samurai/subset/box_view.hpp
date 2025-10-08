// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/box_traverser.hpp"

namespace samurai
{

    template <class B>
    class BoxView;

    template <class B>
    struct SetTraits<BoxView<B>>
    {
        static_assert(std::same_as<Box<typename B::point_t::value_type, B::dim>, B>);

        template <std::size_t>
        using traverser_t = BoxTraverser<B>;

        static constexpr std::size_t dim()
        {
            return B::dim;
        }
    };

    template <class B>
    class BoxView : public SetBase<BoxView<B>>
    {
        using Self = BoxView<B>;

      public:

        SAMURAI_SET_TYPEDEFS

        BoxView(const std::size_t level, const B& box)
            : m_level(level)
            , m_box(box)
        {
        }

        inline std::size_t level_impl() const
        {
            return m_level;
        }

        inline bool exist_impl() const
        {
            return m_box.is_valid();
        }

        inline bool empty_impl() const
        {
            return !exist_impl();
        }

        template <class index_t, std::size_t d>
        inline traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d>) const
        {
            return (m_box.min_corner()[d + 1] <= index[d] && index[d] < m_box.max_corner()[d + 1])
                     ? traverser_t<d>(m_box.min_corner()[d], m_box.max_corner()[d])
                     : traverser_t<d>(0, 0);
        }

        template <std::size_t d>
        constexpr inline void init_get_traverser_work_impl(const std::size_t, std::integral_constant<std::size_t, d>) const
        {
        }

        template <std::size_t d>
        inline void clear_get_traverser_work_impl(std::integral_constant<std::size_t, d>) const
        {
        }

      private:

        std::size_t m_level;
        const B& m_box;
    };

} // namespace samurai
