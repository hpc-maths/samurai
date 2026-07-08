// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../cell_flag.hpp"
#include "../field/concepts.hpp"
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
        SAMURAI_INLINE void operator()(Dim<dim>, T& dest, const T& src) const
        {
            dest(level, i, index) = src(level, i, index);
        }
    };

    template <class T>
        requires field_like<std::remove_cvref_t<T>>
    SAMURAI_INLINE auto copy(T&& dest, T&& src)
    {
        return make_field_operator_function<copy_op>(std::forward<T>(dest), std::forward<T>(src));
    }

    //////////////////////////////
    // tuple_copy_op (batch copy)
    //
    // Copies every (dest, src) pair of fields given as two tuples, in a single
    // traversal of the interval set. The operator is nD by construction: it
    // never calls `field(level, i, index)` (which performs a `find`), but
    // resolves the offset of the first cell of the interval with
    // `mesh.get_index(level, i.start, index...)` (via `memory_offset`) and then
    // walks the interval with a plain loop `ii = 0 .. i.size()`. For vector
    // fields it additionally loops over the components; storage is assumed AoS
    // (cell-major), i.e. the flat buffer index is `cell * n_comp + comp`.
    //////////////////////////////
    template <std::size_t dim, class TInterval>
    class tuple_copy_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(tuple_copy_op)

        // Offset of the first cell of the current interval, for the given mesh.
        template <class Mesh>
        SAMURAI_INLINE std::size_t cell_offset(const Mesh& mesh) const
        {
            return memory_offset(mesh, {level, i.start, index});
        }

        // nD entry point: walk the (dest, src) pairs inside the two tuples.
        // The first_field argument is only used as a type carrier for `dim`
        // and `mesh_t` by the enclosing field_operator_function; it is not
        // accessed here.
        template <class Dsts, class Srcs, class FirstField>
        SAMURAI_INLINE void operator()(Dim<dim>, Dsts& dests, const Srcs& srcs, const FirstField&) const
        {
            const std::size_t off_d = cell_offset(std::get<0>(dests).mesh());
            const std::size_t off_s = cell_offset(std::get<0>(srcs).mesh());
            const std::size_t n     = static_cast<std::size_t>(i.size());

            auto copy_one = [&](auto& dest, const auto& src)
            {
                using Dest = std::remove_cvref_t<decltype(dest)>;
                using Src  = std::remove_cvref_t<decltype(src)>;

                static_assert(Dest::n_comp == Src::n_comp, "tuple_copy: dest and src fields must have the same number of components");
                constexpr std::size_t nc = Dest::n_comp;

                auto* d_data       = dest.data();
                const auto* s_data = src.data();

                if constexpr (nc == 1)
                {
                    for (std::size_t ii = 0; ii < n; ++ii)
                    {
                        d_data[off_d + ii] = s_data[off_s + ii];
                    }
                }
                else if constexpr (detail::is_soa_v<Dest>) // SoA: data()[comp * nb_cells + cell]
                {
                    const std::size_t nb_d = static_cast<std::size_t>(dest.array().shape()[1]);
                    const std::size_t nb_s = static_cast<std::size_t>(src.array().shape()[1]);
                    for (std::size_t ii = 0; ii < n; ++ii)
                    {
                        for (std::size_t c = 0; c < nc; ++c)
                        {
                            d_data[c * nb_d + off_d + ii] = s_data[c * nb_s + off_s + ii];
                        }
                    }
                }
                else // AoS (cell-major): data()[cell * n_comp + comp]
                {
                    for (std::size_t ii = 0; ii < n; ++ii)
                    {
                        for (std::size_t c = 0; c < nc; ++c)
                        {
                            d_data[(off_d + ii) * nc + c] = s_data[(off_s + ii) * nc + c];
                        }
                    }
                }
            };

            std::apply(
                [&](auto&... dest)
                {
                    std::apply(
                        [&](auto&... src)
                        {
                            ((copy_one(dest, src)), ...);
                        },
                        srcs);
                },
                dests);
        }
    };

    // Copy a tuple of destination fields from a tuple of source fields, every
    // pair in a single traversal of the interval set.
    //
    // The first field of dests is passed as an extra argument to
    // make_field_operator_function so that detail::compute_dim<CT...>() and
    // detail::extract_mesh() can find the dimension and the mesh from the
    // argument types (the plain std::tuple arguments carry neither).
    template <class DestTuple, class SrcTuple>
        requires(!field_like<std::remove_cvref_t<DestTuple>> && !field_like<std::remove_cvref_t<SrcTuple>>)
    SAMURAI_INLINE auto copy(DestTuple&& dests, SrcTuple&& srcs)
    {
        return make_field_operator_function<tuple_copy_op>(dests, srcs, std::get<0>(dests));
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
        SAMURAI_INLINE void operator()(Dim<1>, T& tag, std::integral_constant<int, s>) const
        {
            static_nested_loop<1, -s, s + 1>(
                [&](const auto& stencil)
                {
                    tag(level, i + stencil[0]) |= static_cast<std::uint8_t>(CellFlag::keep);
                });
        }

        template <class T, class Flag, int s>
        SAMURAI_INLINE void operator()(Dim<1>, T& tag, const Flag& flag, std::integral_constant<int, s>) const
        {
            auto mask = (tag(level, i) & static_cast<std::uint8_t>(flag));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                static_nested_loop<1, -s, s + 1>(
                                    [&](const auto& stencil)
                                    {
                                        tag(level, i + stencil[0])(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                    });
                            });
        }

        template <class T, int s>
        SAMURAI_INLINE void operator()(Dim<2>, T& tag, std::integral_constant<int, s>) const
        {
            static_nested_loop<2, -s, s + 1>(
                [&](const auto& stencil)
                {
                    tag(level, i + stencil[0], j + stencil[1]) |= static_cast<std::uint8_t>(CellFlag::keep);
                });
        }

        template <class T, class Flag, int s>
        SAMURAI_INLINE void operator()(Dim<2>, T& tag, const Flag& flag, std::integral_constant<int, s>) const
        {
            auto mask = (tag(level, i, j) & static_cast<std::uint8_t>(flag));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                static_nested_loop<2, -s, s + 1>(
                                    [&](const auto& stencil)
                                    {
                                        tag(level, i + stencil[0], j + stencil[1])(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                    });
                            });
        }

        template <class T, int s>
        SAMURAI_INLINE void operator()(Dim<3>, T& tag, std::integral_constant<int, s>) const
        {
            static_nested_loop<3, -s, s + 1>(
                [&](const auto& stencil)
                {
                    tag(level, i + stencil[0], j + stencil[1], k + stencil[2]) |= static_cast<std::uint8_t>(CellFlag::keep);
                });
        }

        template <class T, class Flag, int s>
        SAMURAI_INLINE void operator()(Dim<3>, T& tag, const Flag& flag, std::integral_constant<int, s>) const
        {
            auto mask = (tag(level, i, j, k) & static_cast<std::uint8_t>(flag));

            apply_on_masked(
                mask,
                [&](auto imask)
                {
                    static_nested_loop<3, -s, s + 1>(
                        [&](const auto& stencil)
                        {
                            tag(level, i + stencil[0], j + stencil[1], k + stencil[2])(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                        });
                });
        }
    };

    template <int s, class T>
    SAMURAI_INLINE auto tag_to_keep(T& tag)
    {
        return make_field_operator_function<tag_to_keep_op>(tag, std::integral_constant<int, s>{});
    }

    template <int s, class T, class Flag>
    SAMURAI_INLINE auto tag_to_keep(T& tag, const Flag& flag)
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
        SAMURAI_INLINE void operator()(Dim<1>, T& tag) const
        {
            auto mask = (tag(level + 1, 2 * i) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1) & static_cast<std::uint8_t>(CellFlag::keep));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                tag(level + 1, 2 * i)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<2>, T& tag) const
        {
            auto mask = (tag(level + 1, 2 * i, 2 * j) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i, 2 * j + 1) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j + 1) & static_cast<std::uint8_t>(CellFlag::keep));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                tag(level + 1, 2 * i, 2 * j)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i, 2 * j + 1)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j + 1)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<3>, T& tag) const
        {
            auto mask = (tag(level + 1, 2 * i, 2 * j, 2 * k) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j, 2 * k) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i, 2 * j + 1, 2 * k) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i, 2 * j, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::keep))
                      | (tag(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::keep));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                tag(level + 1, 2 * i, 2 * j, 2 * k)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j, 2 * k)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i, 2 * j + 1, 2 * k)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i, 2 * j, 2 * k + 1)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i, 2 * j + 1, 2 * k + 1)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                                tag(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1)(imask) |= static_cast<std::uint8_t>(CellFlag::keep);
                            });
        }
    };

    template <class T>
    SAMURAI_INLINE auto keep_children_together(T& tag)
    {
        return make_field_operator_function<keep_children_together_op>(tag);
    }

}
