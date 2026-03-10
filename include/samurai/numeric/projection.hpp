// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../operators_base.hpp"

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
                std::array<double, SRC::n_comp> sum{};
                for (std::size_t s = 0; s < src_offsets.size(); ++s)
                {
                    for (std::size_t n = 0; n < SRC::n_comp; ++n)
                    {
                        sum[n] += src_data[src_offsets[s] + i_f * SRC::n_comp + n] + src_data[src_offsets[s] + (i_f + 1) * SRC::n_comp + n];
                    }
                }
                for (std::size_t n = 0; n < SRC::n_comp; ++n)
                {
                    dest_data[dst_offsets + ii * SRC::n_comp + n] = sum[n] * inv;
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
}
