// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../cell_flag.hpp"
#include "../operators_base.hpp"
#include "../static_algorithm.hpp"

namespace samurai
{
    ///////////////////
    // copy operator //
    ///////////////////

    template <std::size_t dim, class TInterval>
    class copy_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(copy_op)

        template <class T>
        inline void operator()(Dim<1>, T& dest, const T& src) const
        {
            dest(level, i) = src(level, i);
        }

        template <class T>
        inline void operator()(Dim<2>, T& dest, const T& src) const
        {
            dest(level, i, j) = src(level, i, j);
        }

        template <class T>
        inline void operator()(Dim<3>, T& dest, const T& src) const
        {
            dest(level, i, j, k) = src(level, i, j, k);
        }
    };

    template <class T>
    inline auto copy(T&& dest, T&& src)
    {
        return make_field_operator_function<copy_op>(std::forward<T>(dest), std::forward<T>(src));
    }

    //////////////////////////
    // tag_to_keep operator //
    //////////////////////////

    template <std::size_t dim, class TInterval>
    class tag_to_keep_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(tag_to_keep_op)

        template <class T, int s>
        inline void operator()(Dim<1>, T& tag, std::integral_constant<int, s>) const
        {
            static_nested_loop<1, -s, s + 1>(
                [&](const auto& stencil)
                {
                    tag(level, i + stencil[0]) |= static_cast<int>(CellFlag::keep);
                });
        }

        template <class T, class Flag, int s>
        inline void operator()(Dim<1>, T& tag, const Flag& flag, std::integral_constant<int, s>) const
        {
            auto mask = (tag(level, i) & static_cast<int>(flag));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                static_nested_loop<1, -s, s + 1>(
                                    [&](const auto& stencil)
                                    {
                                        tag(level, i + stencil[0])(imask) |= static_cast<int>(CellFlag::keep);
                                    });
                            });
        }

        template <class T, int s>
        inline void operator()(Dim<2>, T& tag, std::integral_constant<int, s>) const
        {
            static_nested_loop<2, -s, s + 1>(
                [&](const auto& stencil)
                {
                    tag(level, i + stencil[0], j + stencil[1]) |= static_cast<int>(CellFlag::keep);
                });
        }

        template <class T, class Flag, int s>
        inline void operator()(Dim<2>, T& tag, const Flag& flag, std::integral_constant<int, s>) const
        {
            auto mask = (tag(level, i, j) & static_cast<int>(flag));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                static_nested_loop<2, -s, s + 1>(
                                    [&](const auto& stencil)
                                    {
                                        tag(level, i + stencil[0], j + stencil[1])(imask) |= static_cast<int>(CellFlag::keep);
                                    });
                            });
        }

        template <class T, int s>
        inline void operator()(Dim<3>, T& tag, std::integral_constant<int, s>) const
        {
            static_nested_loop<3, -s, s + 1>(
                [&](const auto& stencil)
                {
                    tag(level, i + stencil[0], j + stencil[1], k + stencil[2]) |= static_cast<int>(CellFlag::keep);
                });
        }

        template <class T, class Flag, int s>
        inline void operator()(Dim<3>, T& tag, const Flag& flag, std::integral_constant<int, s>) const
        {
            auto mask = (tag(level, i, j, k) & static_cast<int>(flag));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                static_nested_loop<3, -s, s + 1>(
                                    [&](const auto& stencil)
                                    {
                                        tag(level, i + stencil[0], j + stencil[1], k + stencil[2])(imask) |= static_cast<int>(CellFlag::keep);
                                    });
                            });
        }
    };

    template <int s, class T>
    inline auto tag_to_keep(T& tag)
    {
        return make_field_operator_function<tag_to_keep_op>(tag, std::integral_constant<int, s>{});
    }

    template <int s, class T, class Flag>
    inline auto tag_to_keep(T& tag, const Flag& flag)
    {
        return make_field_operator_function<tag_to_keep_op>(tag, flag, std::integral_constant<int, s>{});
    }

    /////////////////////////////////////
    // keep_children_together operator //
    /////////////////////////////////////

    template <std::size_t dim, class TInterval>
    class keep_children_together_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(keep_children_together_op)

        template <class T>
        inline void operator()(Dim<1>, T& tag) const
        {
            auto mask = (tag(level + 1, 2 * i) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1) & static_cast<int>(CellFlag::keep));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                tag(level + 1, 2 * i)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1)(imask) |= static_cast<int>(CellFlag::keep);
                            });
        }

        template <class T>
        inline void operator()(Dim<2>, T& tag) const
        {
            auto mask = (tag(level + 1, 2 * i, 2 * j) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::keep));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                tag(level + 1, 2 * i, 2 * j)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i, 2 * j + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j + 1)(imask) |= static_cast<int>(CellFlag::keep);
                            });
        }

        template <class T>
        inline void operator()(Dim<3>, T& tag) const
        {
            auto mask = (tag(level + 1, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                tag(level + 1, 2 * i, 2 * j, 2 * k)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j, 2 * k)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i, 2 * j + 1, 2 * k)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i, 2 * j, 2 * k + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i, 2 * j + 1, 2 * k + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1)(imask) |= static_cast<int>(CellFlag::keep);
                            });
        }
    };

    template <class T>
    inline auto keep_children_together(T& tag)
    {
        return make_field_operator_function<keep_children_together_op>(tag);
    }

}
