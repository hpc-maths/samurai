// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../operators_base.hpp"
#include "../static_algorithm.hpp"

namespace samurai
{
    /////////////////////////
    // projection operator //
    /////////////////////////

    template <std::size_t dim, class TInterval>
    class projection_op_ : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(projection_op_)

        template <class DEST, class SRC>
        SAMURAI_INLINE void operator()(Dim<dim>, DEST& dest, const SRC& src) const
        {
            static_assert(DEST::n_comp == SRC::n_comp, "Source and destination fields must have the same number of components");
            auto dst_offsets = memory_offset(dest.mesh(), {level, i.start, index});

            std::array<std::size_t, 1ULL << (dim - 1)> src_offsets;
            if constexpr (dim == 1)
            {
                src_offsets[0] = memory_offset(src.mesh(), {level + 1, 2 * i.start});
            }
            else
            {
                std::size_t ind = 0;
                static_nested_loop<dim - 1, 0, 2>(
                    [&](const auto& stencil)
                    {
                        auto new_index     = 2 * index + stencil;
                        src_offsets[ind++] = memory_offset(src.mesh(), {level + 1, 2 * i.start, new_index});
                    });
            }

            const auto* src_data = src.data();
            auto* dest_data      = dest.data();
            constexpr double inv = 1.0 / static_cast<double>(1ULL << dim);

            for (std::size_t ii = 0, i_f = 0; ii < i.size(); ++ii, i_f += 2)
            {
                std::array<double, SRC::n_comp> sum;
                sum.fill(0);
                for (std::size_t s = 0; s < src_offsets.size(); ++s)
                {
                    for (std::size_t n = 0; n < SRC::n_comp; ++n)
                    {
                        sum[n] += src_data[(src_offsets[s] + i_f) * SRC::n_comp + n] + src_data[(src_offsets[s] + i_f + 1) * SRC::n_comp + n];
                    }
                }
                for (std::size_t n = 0; n < SRC::n_comp; ++n)
                {
                    dest_data[(dst_offsets + ii) * SRC::n_comp + n] = sum[n] * inv;
                }
            }
        }
    };

    template <std::size_t dim, class TInterval>
    class variadic_projection_op_ : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(variadic_projection_op_)

        template <std::size_t d>
        SAMURAI_INLINE void operator()(Dim<d>) const
        {
        }

        template <class Head, class... Tail>
        SAMURAI_INLINE void operator()(Dim<1>, Head& source, Tail&... sources) const
        {
            projection_op_<dim, interval_t>(level, i)(Dim<1>{}, source, source);
            this->operator()(Dim<1>{}, sources...);
        }

        template <class Head, class... Tail>
        SAMURAI_INLINE void operator()(Dim<2>, Head& source, Tail&... sources) const
        {
            projection_op_<dim, interval_t>(level, i, j)(Dim<2>{}, source, source);
            this->operator()(Dim<2>{}, sources...);
        }

        template <class Head, class... Tail>
        SAMURAI_INLINE void operator()(Dim<3>, Head& source, Tail&... sources) const
        {
            projection_op_<dim, interval_t>(level, i, j, k)(Dim<3>{}, source, source);
            this->operator()(Dim<3>{}, sources...);
        }
    };

    // Tuple-based projection: projects pairs (dest, src) in a single traversal
    template <std::size_t dim, class TInterval>
    class tuple_projection_op_ : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(tuple_projection_op_)

        // Offset of the first cell of the current interval, for the given mesh.
        template <class Mesh>
        SAMURAI_INLINE std::size_t cell_offset(const Mesh& mesh) const
        {
            if constexpr (dim == 1)
            {
                return memory_offset(mesh, {level, i.start});
            }
            else
            {
                return memory_offset(mesh, {level, i.start, index});
            }
        }

        // Project one (dest, src) field pair over the current interval.
        template <class Dest, class Src>
        SAMURAI_INLINE void project_one(Dest& dest, const Src& src) const
        {
            static_assert(Dest::n_comp == Src::n_comp, "Source and destination fields must have the same number of components");

            const std::size_t off_d = cell_offset(dest.mesh());
            constexpr std::size_t nc = Dest::n_comp;
            const std::size_t n = static_cast<std::size_t>(i.size());

            std::array<std::size_t, 1ULL << (dim - 1)> src_offsets;
            if constexpr (dim == 1)
            {
                src_offsets[0] = memory_offset(src.mesh(), {level + 1, 2 * i.start});
            }
            else
            {
                std::size_t ind = 0;
                static_nested_loop<dim - 1, 0, 2>(
                    [&](const auto& stencil)
                    {
                        auto new_index     = 2 * index + stencil;
                        src_offsets[ind++] = memory_offset(src.mesh(), {level + 1, 2 * i.start, new_index});
                    });
            }

            const auto* src_data = src.data();
            auto* dest_data      = dest.data();
            constexpr double inv = 1.0 / static_cast<double>(1ULL << dim);

            for (std::size_t ii = 0, i_f = 0; ii < n; ++ii, i_f += 2)
            {
                std::array<double, nc> sum;
                sum.fill(0);
                for (std::size_t s = 0; s < src_offsets.size(); ++s)
                {
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        sum[c] += src_data[(src_offsets[s] + i_f) * nc + c] + src_data[(src_offsets[s] + i_f + 1) * nc + c];
                    }
                }
                for (std::size_t c = 0; c < nc; ++c)
                {
                    dest_data[(off_d + ii) * nc + c] = sum[c] * inv;
                }
            }
        }

        // nD entry point: walk the interleaved (dest, src, dest, src, ...) pack
        // one pair at a time.
        template <class DHead, class SHead, class... Tail>
        SAMURAI_INLINE void operator()(Dim<dim>, DHead& dest_head, const SHead& src_head, Tail&... tail) const
        {
            project_one(dest_head, src_head);
            if constexpr (sizeof...(Tail) > 0)
            {
                (*this)(Dim<dim>{}, tail...);
            }
        }
    };

    template <class T>
    SAMURAI_INLINE auto projection(T&& field)
    {
        return make_field_operator_function<projection_op_>(std::forward<T>(field), std::forward<T>(field));
    }

    template <class... T>
    SAMURAI_INLINE auto variadic_projection(T&&... fields)
    {
        return make_field_operator_function<variadic_projection_op_>(std::forward<T>(fields)...);
    }

    template <class T1, class T2>
    SAMURAI_INLINE auto projection(T1&& field_dest, T2&& field_src)
    {
        return make_field_operator_function<projection_op_>(std::forward<T1>(field_dest), std::forward<T2>(field_src));
    }

    namespace detail
    {
        template <class DestTuple, class SrcTuple, std::size_t... Is>
        SAMURAI_INLINE auto make_tuple_projection(DestTuple& dests, SrcTuple& srcs, std::index_sequence<Is...>)
        {
            // Build the interleaved argument list (dest0, src0, dest1, src1, ...).
            auto interleaved = std::tuple_cat(std::tie(std::get<Is>(dests), std::get<Is>(srcs))...);
            return std::apply(
                [](auto&... args)
                {
                    return make_field_operator_function<tuple_projection_op_>(args...);
                },
                interleaved);
        }
    }

    // Project a tuple of destination fields from a tuple of source fields
    template <class DestTuple, class SrcTuple>
        requires(!field_like<std::remove_cvref_t<DestTuple>> && !field_like<std::remove_cvref_t<SrcTuple>>)
    SAMURAI_INLINE auto projection(DestTuple&& dests, SrcTuple&& srcs)
    {
        auto& dt = detail::tuple_refs(dests);
        auto& st = detail::tuple_refs(srcs);
        constexpr std::size_t n = std::tuple_size_v<std::remove_cvref_t<decltype(dt)>>;
        static_assert(n == std::tuple_size_v<std::remove_cvref_t<decltype(st)>>,
                      "projection(tuples): the dest and src tuples must contain the same number of fields");
        return detail::make_tuple_projection(dt, st, std::make_index_sequence<n>{});
    }
}
