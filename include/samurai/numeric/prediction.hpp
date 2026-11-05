// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <xtensor/views/xview.hpp>

#include "../field/concepts.hpp"
#include "../operators_base.hpp"
#include "../static_algorithm.hpp"
#include "../utils.hpp"
#include "prediction_coefficients.hpp"

namespace samurai
{
    // Superseded by @ref prediction_coefficients (numeric/prediction_coefficients.hpp), which
    // solves these same values from the cell-average moment conditions instead of tabulating
    // them, and generalises them to the boundary-shifted stencils. Nothing in the library calls
    // interp_coeffs any more; it is kept as the independent reference that the generated family
    // is checked against, bit for bit, in tests/test_prediction_coefficients.cpp. It goes away
    // with the public API break of the boundary rewrite, along with the order ceiling it imposes.
    template <std::size_t s>
    SAMURAI_INLINE std::array<double, s> interp_coeffs(double sign);

    template <>
    SAMURAI_INLINE std::array<double, 1> interp_coeffs(double)
    {
        return {1};
    }

    template <>
    SAMURAI_INLINE std::array<double, 3> interp_coeffs(double sign)
    {
        return {sign / 8., 1, -sign / 8.};
    }

    template <>
    SAMURAI_INLINE std::array<double, 5> interp_coeffs(double sign)
    {
        return {-sign * 3. / 128., sign * 22. / 128., 1, -sign * 22 / 128., sign * 3. / 128.};
    }

    template <>
    SAMURAI_INLINE std::array<double, 7> interp_coeffs(double sign)
    {
        return {sign * 5. / 1024., -sign * 11. / 256., sign * 201. / 1024., 1, -sign * 201. / 1024., sign * 11. / 256., -sign * 5. / 1024.};
    }

    template <>
    SAMURAI_INLINE std::array<double, 9> interp_coeffs(double sign)
    {
        return {-sign * 35. / 32768.,
                sign * 185. / 16384,
                -sign * 949. / 16384,
                sign * 3461. / 16384.,
                1,
                -sign * 3461. / 16384.,
                sign * 949. / 16384,
                -sign * 185. / 16384,
                sign * 35. / 32768.};
    }

    template <>
    SAMURAI_INLINE std::array<double, 11> interp_coeffs(double sign)
    {
        return {sign * 63. / 262144.,
                -sign * 49. / 16384.,
                sign * 4661. / 262144.,
                -sign * 569. / 8192.,
                sign * 29011. / 131072.,
                1,
                -sign * 29011. / 131072.,
                sign * 569. / 8192.,
                -sign * 4661. / 262144.,
                sign * 49. / 16384.,
                -sign * 63. / 262144.};
    }

    namespace detail
    {
        /**
         * Number of coarse cells a prediction kernel processes at once. Its buffers hold
         * @a rows rows of @c (chunk + 2 * radius) cells of @a n_comp values; the chunk is
         * sized so that they stay around 128 KiB whatever the dimension.
         */
        template <std::size_t rows, std::size_t radius, std::size_t n_comp>
        consteval std::size_t prediction_chunk()
        {
            constexpr std::size_t budget = 16384 / (rows * n_comp);
            constexpr std::size_t chunk  = budget > 2 * radius ? budget - 2 * radius : 0;
            return chunk < 4 ? 4 : (chunk > 64 ? 64 : chunk);
        }

        /// A fixed-size array when it is small, a heap one otherwise: the stencil of a
        /// high dimension has too many rows for the stack.
        template <class T, std::size_t N>
        struct prediction_buffer
        {
            static constexpr bool on_stack = N <= 2048;
            std::conditional_t<on_stack, std::array<T, on_stack ? N : 1>, std::vector<T>> data;

            prediction_buffer()
            {
                if constexpr (!on_stack)
                {
                    data.resize(N);
                }
            }

            T* get()
            {
                return data.data();
            }
        };

        /**
         * Memory offsets of the rows a prediction stencil reads: the coarse cells
         * @c (x_lo, parent + k) for every @c k in @c [-radius, radius]^(dim-1), in the order
         * where the outermost direction varies slowest.
         *
         * Each direction is searched once per value of the directions above it, instead of
         * once per stencil cell, and the cells along x are not searched at all: the rows
         * are contiguous from @a x_lo to @a x_hi, which the caller relies on.
         */
        template <std::size_t N, std::size_t radius, std::size_t dim, class TInterval, class Index>
        SAMURAI_INLINE void prediction_row_offsets(const LevelCellArray<dim, TInterval>& lca,
                                                   std::size_t start,
                                                   std::size_t end,
                                                   typename TInterval::value_t x_lo,
                                                   [[maybe_unused]] typename TInterval::value_t x_hi,
                                                   const Index& parent,
                                                   std::size_t*& out)
        {
            using value_t = typename TInterval::value_t;
            using diff_t  = std::ptrdiff_t;

            if constexpr (N == 0)
            {
                const auto pos = interval_search(lca[0].cbegin() + static_cast<diff_t>(start),
                                                 lca[0].cbegin() + static_cast<diff_t>(end),
                                                 x_lo);
                assert(pos >= 0 && "prediction: a cell of the stencil is missing");
                const auto& x_interval = lca[0][start + static_cast<std::size_t>(pos)];
                assert(x_interval.contains(x_hi - 1) && "prediction: a row of the stencil is not contiguous");
                *out++ = static_cast<std::size_t>(x_interval.index + x_lo);
            }
            else
            {
                for (value_t k = -static_cast<value_t>(radius); k <= static_cast<value_t>(radius); ++k)
                {
                    const value_t y = parent[N - 1] + k;
                    const auto pos  = interval_search(lca[N].cbegin() + static_cast<diff_t>(start),
                                                     lca[N].cbegin() + static_cast<diff_t>(end),
                                                     y);
                    assert(pos >= 0 && "prediction: a cell of the stencil is missing");
                    const auto off = static_cast<std::size_t>(lca[N][start + static_cast<std::size_t>(pos)].index + y);
                    prediction_row_offsets<N - 1, radius>(lca, lca.offsets(N)[off], lca.offsets(N)[off + 1], x_lo, x_hi, parent, out);
                }
            }
        }

        /**
         * Contracts directions @c 1..N of the stencil for both children in each of them,
         * over one chunk of @a len values per row.
         *
         * @a out receives @c 2^N rows; row @c q is the child whose parity in direction @c j
         * is bit @c j-1 of @c q. Each direction is contracted once, from the innermost
         * outwards, so the cost grows like @c order^N instead of the @c (2*order)^N of the
         * full tensor product. @a scratch must hold @c 2^N - 1 more rows.
         */
        template <std::size_t N, std::size_t order, std::size_t n_comp, class Value>
        SAMURAI_INLINE void contract_family(const Value* src,
                                            const std::size_t*& row_offset,
                                            std::size_t shift,
                                            std::size_t len,
                                            std::size_t stride,
                                            const std::array<std::array<double, order>, 2>& c,
                                            double* out,
                                            double* scratch)
        {
            constexpr std::size_t n_out = std::size_t{1} << N;
            for (std::size_t q = 0; q < n_out; ++q)
            {
                std::fill_n(out + q * stride, len, 0.);
            }

            for (std::size_t k = 0; k < order; ++k)
            {
                if constexpr (N == 1)
                {
                    const Value* src_row = src + (*row_offset++) * n_comp + shift;
                    const double c0      = c[0][k];
                    const double c1      = c[1][k];
                    for (std::size_t m = 0; m < len; ++m)
                    {
                        out[m] += c0 * src_row[m];
                        out[stride + m] += c1 * src_row[m];
                    }
                }
                else
                {
                    constexpr std::size_t n_sub = n_out / 2;
                    contract_family<N - 1, order, n_comp>(src, row_offset, shift, len, stride, c, scratch, scratch + n_sub * stride);
                    for (std::size_t q = 0; q < 2; ++q)
                    {
                        const double ck = c[q][k];
                        for (std::size_t sub = 0; sub < n_sub; ++sub)
                        {
                            double* o       = out + (q * n_sub + sub) * stride;
                            const double* t = scratch + sub * stride;
                            for (std::size_t m = 0; m < len; ++m)
                            {
                                o[m] += ck * t[m];
                            }
                        }
                    }
                }
            }
        }
    }

    /////////////////////////
    // prediction operator //
    /////////////////////////

    template <std::size_t dim, class TInterval>
    class prediction_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(prediction_op)

        template <class DEST, class SRC>
        void
        operator()(Dim<dim>, DEST& dest, const SRC& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, true>) const;

        template <class DEST, class SRC>
        void
        operator()(Dim<dim>, DEST& dest, const SRC& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, false>) const;

        template <class DEST, class SRC, std::size_t order>
        void
        operator()(Dim<dim>, DEST& dest, const SRC& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, true>) const;

        template <class DEST, class SRC, std::size_t order>
        void
        operator()(Dim<dim>, DEST& dest, const SRC& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, false>) const;
    };

    template <std::size_t dim, class TInterval>
    template <class DEST, class SRC>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<dim>,
                                                                  DEST& dest,
                                                                  const SRC& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, true>) const
    {
        static_assert(DEST::n_comp == SRC::n_comp, "Source and destination fields must have the same number of components");

        auto src_offset = memory_offset(src.mesh(), {level, i.start, index});

        std::vector<std::size_t> dest_offsets;
        dest_offsets.reserve(1ULL << (dim - 1));

        static_nested_loop<dim - 1, 0, 2>(
            [&](const auto& stencil)
            {
                auto new_index = 2 * index + stencil;
                dest_offsets.push_back(memory_offset(dest.mesh(), {level + 1, 2 * i.start, new_index}));
            });

        const auto* src_data = src.data();
        auto* dest_data      = dest.data();

        for (std::size_t i_c = 0, i_f = 0; i_c < i.size(); ++i_c, i_f += 2)
        {
            for (std::size_t s = 0; s < dest_offsets.size(); ++s)
            {
                const std::size_t src_index   = (src_offset + i_c) * SRC::n_comp;
                const std::size_t dest_index0 = (dest_offsets[s] + i_f) * SRC::n_comp;
                const std::size_t dest_index1 = (dest_offsets[s] + i_f + 1) * SRC::n_comp;

                for (std::size_t n = 0; n < SRC::n_comp; ++n)
                {
                    dest_data[dest_index0 + n] = src_data[src_index + n];
                    dest_data[dest_index1 + n] = src_data[src_index + n];
                }
            }
        }
    }

    template <std::size_t dim, class TInterval>
    template <class DEST, class SRC, std::size_t pred_stencil_size>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<dim>,
                                                                  DEST& dest,
                                                                  const SRC& src,
                                                                  std::integral_constant<std::size_t, pred_stencil_size>,
                                                                  std::integral_constant<bool, true>) const
    {
        // Predicts the 2^dim children of every cell of the interval i at `level` onto
        // `level + 1`. The tensor product is contracted one direction at a time: first the
        // directions above x, for every child row at once (detail::contract_family), then x.
        static_assert(DEST::n_comp == SRC::n_comp, "Source and destination fields must have the same number of components");

        using value_t                  = typename TInterval::value_t;
        using mesh_id_t                = typename SRC::mesh_t::mesh_id_t;
        constexpr std::size_t r        = pred_stencil_size;
        constexpr std::size_t order    = 2 * r + 1;
        constexpr std::size_t n_comp   = SRC::n_comp;
        constexpr std::size_t n_rows   = ce_pow(order, dim - 1);
        constexpr std::size_t n_family = std::size_t{1} << (dim - 1);
        constexpr std::size_t chunk    = detail::prediction_chunk<2 * n_family, r, n_comp>();
        constexpr std::size_t stride   = (chunk + 2 * r) * n_comp;

        // (even index coefficients, odd index coefficients)
        const std::array<std::array<double, order>, 2> c = {
            {prediction_coefficients<r>(0, 0).c, prediction_coefficients<r>(1, 0).c}
        };

        const value_t cs = i.start;
        const value_t ce = i.end;

        detail::prediction_buffer<std::size_t, n_rows> row_offsets;
        {
            const auto& lca  = src.mesh()[mesh_id_t::reference][level];
            std::size_t* out = row_offsets.get();
            detail::prediction_row_offsets<dim - 1, r>(lca,
                                                       0,
                                                       lca[dim - 1].size(),
                                                       cs - static_cast<value_t>(r),
                                                       ce + static_cast<value_t>(r),
                                                       index,
                                                       out);
        }

        // Child row q is the row 2 * index + (bit j-1 of q in direction j).
        std::array<std::size_t, n_family> dest_offsets;
        for (std::size_t q = 0; q < n_family; ++q)
        {
            xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> child = 2 * index;
            for (std::size_t d = 1; d < dim; ++d)
            {
                child[d - 1] += static_cast<value_t>((q >> (d - 1)) & 1);
            }
            dest_offsets[q] = memory_offset(dest.mesh(), {level + 1, 2 * cs, child});
        }

        const auto* src_data = src.data();
        auto* dest_data      = dest.data();

        // In 1D there is no direction above x to contract: the x contraction reads the
        // source row directly and the buffer of contracted rows is empty.
        detail::prediction_buffer<double, dim == 1 ? 0 : 2 * n_family * stride> buffer;

        for (value_t a = cs; a < ce; a += static_cast<value_t>(chunk))
        {
            const value_t b         = std::min(ce, a + static_cast<value_t>(chunk));
            const std::size_t shift = static_cast<std::size_t>(a - cs) * n_comp;

            // Contracts x for the child row q, from the row t contracted above x.
            auto contract_x = [&](std::size_t q, const auto* t)
            {
                for (value_t xc = a; xc < b; ++xc)
                {
                    const auto x      = static_cast<std::size_t>(xc - a);
                    const auto fine_x = 2 * static_cast<std::size_t>(xc - cs);
                    for (std::size_t p = 0; p < 2; ++p)
                    {
                        auto* d = dest_data + (dest_offsets[q] + fine_x + p) * n_comp;
                        for (std::size_t n = 0; n < n_comp; ++n)
                        {
                            double v = 0.;
                            for (std::size_t s = 0; s < order; ++s)
                            {
                                v += c[p][s] * t[(x + s) * n_comp + n];
                            }
                            d[n] = v;
                        }
                    }
                }
            };

            if constexpr (dim == 1)
            {
                contract_x(0, src_data + row_offsets.get()[0] * n_comp + shift);
            }
            else
            {
                const std::size_t len         = static_cast<std::size_t>(b - a + 2 * static_cast<value_t>(r)) * n_comp;
                const std::size_t* row_offset = row_offsets.get();
                double* rows                  = buffer.get();
                detail::contract_family<dim - 1, order, n_comp>(src_data, row_offset, shift, len, stride, c, rows, rows + n_family * stride);
                for (std::size_t q = 0; q < n_family; ++q)
                {
                    contract_x(q, rows + q * stride);
                }
            }
        }
    }

    template <std::size_t dim, class TInterval>
    template <class DEST, class SRC>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<dim>,
                                                                  DEST& dest,
                                                                  const SRC& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, false>) const
    {
        static_assert(DEST::n_comp == SRC::n_comp, "Source and destination fields must have the same number of components");
        using value_t = typename TInterval::value_t;

        const auto* src_data = src.data();
        auto* dest_data      = dest.data();

        auto src_offset  = memory_offset(src.mesh(), {level - 1, i.start >> 1, index >> 1});
        auto dest_offset = memory_offset(dest.mesh(), {level, i.start, index});

        for (std::size_t i_f = 0; i_f < i.size(); ++i_f)
        {
            const std::size_t i_c = static_cast<std::size_t>(((i.start + static_cast<value_t>(i_f)) >> 1) - (i.start >> 1));
            for (std::size_t n = 0; n < SRC::n_comp; ++n)
            {
                dest_data[(dest_offset + i_f) * SRC::n_comp + n] = src_data[(src_offset + i_c) * SRC::n_comp + n];
            }
        }
    }

    template <std::size_t dim, class TInterval>
    template <class DEST, class SRC, std::size_t pred_stencil_size>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<dim>,
                                                                  DEST& dest,
                                                                  const SRC& src,
                                                                  std::integral_constant<std::size_t, pred_stencil_size>,
                                                                  std::integral_constant<bool, false>) const
    {
        // Predicts the cells of the interval i at `level` from their parents at
        // `level - 1`. Above x, every cell of the interval has the same parity, so the
        // stencil rows are first summed with fixed weights; the x direction comes last.
        static_assert(DEST::n_comp == SRC::n_comp, "Source and destination fields must have the same number of components");

        using value_t                = typename TInterval::value_t;
        using mesh_id_t              = typename SRC::mesh_t::mesh_id_t;
        constexpr std::size_t r      = pred_stencil_size;
        constexpr std::size_t order  = 2 * r + 1;
        constexpr std::size_t n_comp = SRC::n_comp;
        constexpr std::size_t n_rows = ce_pow(order, dim - 1);
        constexpr std::size_t chunk  = detail::prediction_chunk<1, r, n_comp>();

        // (even index coefficients, odd index coefficients)
        const std::array<std::array<double, order>, 2> c = {
            {prediction_coefficients<r>(0, 0).c, prediction_coefficients<r>(1, 0).c}
        };

        const value_t cs = i.start >> 1;
        const value_t ce = ((i.end - 1) >> 1) + 1;

        auto parent = index;
        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            parent[d] = index[d] >> 1;
        }

        detail::prediction_buffer<std::size_t, n_rows> row_offsets;
        {
            const auto& lca  = src.mesh()[mesh_id_t::reference][level - 1];
            std::size_t* out = row_offsets.get();
            detail::prediction_row_offsets<dim - 1, r>(lca,
                                                       0,
                                                       lca[dim - 1].size(),
                                                       cs - static_cast<value_t>(r),
                                                       ce + static_cast<value_t>(r),
                                                       parent,
                                                       out);
        }

        // Weight of each row: the product of the coefficients of its directions above x,
        // in the order of prediction_row_offsets (outermost direction slowest).
        // In 1D there is no direction above x: the single row has weight 1 and the
        // x contraction reads the source row directly, so neither the weights nor the
        // summed row t are needed.
        detail::prediction_buffer<double, dim == 1 ? 0 : n_rows> weights;
        if constexpr (dim > 1)
        {
            double* w     = weights.get();
            w[0]          = 1.;
            std::size_t n = 1;
            for (std::size_t d = dim - 1; d >= 1; --d)
            {
                const auto& cd = c[static_cast<std::size_t>(index[d - 1] & 1)];
                for (std::size_t row_id = n; row_id-- > 0;)
                {
                    for (std::size_t s = order; s-- > 0;)
                    {
                        w[row_id * order + s] = w[row_id] * cd[s];
                    }
                }
                n *= order;
            }
        }

        const auto* src_data = src.data();
        auto* dest_data      = dest.data();
        auto dest_offset     = memory_offset(dest.mesh(), {level, i.start, index});

        std::array<double, dim == 1 ? 0 : (chunk + 2 * r) * n_comp> t;

        for (value_t a = cs; a < ce; a += static_cast<value_t>(chunk))
        {
            const value_t b         = std::min(ce, a + static_cast<value_t>(chunk));
            const std::size_t shift = static_cast<std::size_t>(a - cs) * n_comp;
            const auto* src_row     = src_data + row_offsets.get()[0] * n_comp + shift;

            // Contracts x from the row tx, already summed above x.
            auto contract_x = [&](const auto* tx)
            {
                const value_t f_begin = std::max(i.start, 2 * a);
                const value_t f_end   = std::min(i.end, 2 * b);
                for (value_t f = f_begin; f < f_end; ++f)
                {
                    const auto x   = static_cast<std::size_t>((f >> 1) - a);
                    const auto& cp = c[static_cast<std::size_t>(f & 1)];
                    auto* d        = dest_data + (dest_offset + static_cast<std::size_t>(f - i.start)) * n_comp;
                    for (std::size_t n = 0; n < n_comp; ++n)
                    {
                        double v = 0.;
                        for (std::size_t s = 0; s < order; ++s)
                        {
                            v += cp[s] * tx[(x + s) * n_comp + n];
                        }
                        d[n] = v;
                    }
                }
            };

            if constexpr (dim == 1)
            {
                contract_x(src_row);
            }
            else
            {
                const std::size_t len = static_cast<std::size_t>(b - a + 2 * static_cast<value_t>(r)) * n_comp;
                const double w0       = weights.get()[0];
                for (std::size_t m = 0; m < len; ++m)
                {
                    t[m] = w0 * src_row[m];
                }
                for (std::size_t row_id = 1; row_id < n_rows; ++row_id)
                {
                    src_row         = src_data + row_offsets.get()[row_id] * n_comp + shift;
                    const double wr = weights.get()[row_id];
                    for (std::size_t m = 0; m < len; ++m)
                    {
                        t[m] += wr * src_row[m];
                    }
                }
                contract_x(t.data());
            }
        }
    }

    template <std::size_t dim, class TInterval>
    class variadic_prediction_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(variadic_prediction_op)

        template <std::size_t d, std::size_t order, bool dest_on_level>
        SAMURAI_INLINE void operator()(Dim<d>, std::integral_constant<std::size_t, order>, std::integral_constant<bool, dest_on_level>) const
        {
        }

        template <std::size_t order, bool dest_on_level, class Head, class... Tail>
        SAMURAI_INLINE void operator()(Dim<dim>,
                                       std::integral_constant<std::size_t, order> o,
                                       std::integral_constant<bool, dest_on_level> dest,
                                       Head& source,
                                       Tail&... sources) const
        {
            prediction_op<dim, interval_t>(level, i, index)(Dim<dim>{}, source, source, o, dest);
            this->operator()(Dim<dim>{}, o, dest, sources...);
        }
    };

    // Tuple-based prediction: predicts pairs (dest, src) given as two tuples
    // in a single traversal of the interval set.
    template <std::size_t dim, class TInterval>
    class tuple_prediction_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(tuple_prediction_op)

        // Predict one (dest, src) field pair over the current interval.
        // Uses the base prediction_op for each pair.
        template <std::size_t order, bool dest_on_level, class Dest, class Src>
        SAMURAI_INLINE void predict_one(Dest& dest, const Src& src) const
        {
            prediction_op<dim, interval_t>(
                level,
                i,
                index)(Dim<dim>{}, dest, src, std::integral_constant<std::size_t, order>{}, std::integral_constant<bool, dest_on_level>{});
        }

        // nD entry point: walk the (dest, src) pairs inside the two tuples.
        // The first_field argument is only used as a type carrier for `dim`
        // and `mesh_t` by the enclosing field_operator_function.
        template <std::size_t order, bool dest_on_level, class Dsts, class Srcs, class FirstField>
        SAMURAI_INLINE void operator()(Dim<dim>,
                                       std::integral_constant<std::size_t, order>,
                                       std::integral_constant<bool, dest_on_level>,
                                       Dsts& dests,
                                       const Srcs& srcs,
                                       const FirstField&) const
        {
            std::apply(
                [&](auto&... dest)
                {
                    std::apply(
                        [&](auto&... src)
                        {
                            ((predict_one<order, dest_on_level>(dest, src)), ...);
                        },
                        srcs);
                },
                dests);
        }
    };

    template <std::size_t order, bool dest_on_level, class... T>
    SAMURAI_INLINE auto variadic_prediction(T&&... fields)
    {
        return make_field_operator_function<variadic_prediction_op>(std::integral_constant<std::size_t, order>{},
                                                                    std::integral_constant<bool, dest_on_level>{},
                                                                    std::forward<T>(fields)...);
    }

    template <std::size_t order, bool dest_on_level, class T>
    SAMURAI_INLINE auto prediction(T& field)
    {
        return make_field_operator_function<prediction_op>(field,
                                                           field,
                                                           std::integral_constant<std::size_t, order>{},
                                                           std::integral_constant<bool, dest_on_level>{});
    }

    template <std::size_t order, bool dest_on_level, class DEST, class SRC>
    SAMURAI_INLINE auto prediction(DEST& field_dest, const SRC& field_src)
    {
        return make_field_operator_function<prediction_op>(field_dest,
                                                           field_src,
                                                           std::integral_constant<std::size_t, order>{},
                                                           std::integral_constant<bool, dest_on_level>{});
    }

    // Predict a tuple of destination fields from a tuple of source fields,
    // every pair in a single traversal of the interval set.
    //
    // The first field of dests is passed as an extra argument to
    // make_field_operator_function so that detail::compute_dim<CT...>() and
    // detail::extract_mesh() can find the dimension and the mesh from the
    // argument types (the plain std::tuple arguments carry neither).
    template <std::size_t order, bool dest_on_level, class DestTuple, class SrcTuple>
        requires(!field_like<std::remove_cvref_t<DestTuple>> && !field_like<std::remove_cvref_t<SrcTuple>>)
    SAMURAI_INLINE auto prediction(DestTuple&& dests, SrcTuple&& srcs)
    {
        constexpr std::size_t n = std::tuple_size_v<std::remove_cvref_t<DestTuple>>;
        static_assert(n == std::tuple_size_v<std::remove_cvref_t<SrcTuple>>,
                      "prediction(tuples): the dest and src tuples must contain the same number of fields");
        return make_field_operator_function<tuple_prediction_op>(std::integral_constant<std::size_t, order>{},
                                                                 std::integral_constant<bool, dest_on_level>{},
                                                                 dests,
                                                                 srcs,
                                                                 std::get<0>(dests));
    }
}
