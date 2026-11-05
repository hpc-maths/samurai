// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

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
         * The prediction stencil of one coarse row, contracted one direction at a time.
         *
         * A predicted value sums the coarse cells @c parent + k, @c k in @c [-r, r]^dim, each
         * weighted by the product over the directions @c d of @c c[parity_d][k_d]. The sum is
         * separable, so the directions above x are contracted one after the other: contracting
         * direction @c N combines the @c 2r+1 contractions of direction @c N-1 into one row per
         * parity of direction @c N, and x comes last.
         *
         * @a n_par is the number of parities kept in every direction above x: 1 to predict a
         * single fine row, 2 to predict the @c 2^(dim-1) rows of children of a coarse row at
         * once. With 2, each contraction of direction @c N-1 serves both children of direction
         * @c N, and the cost grows like @c (2r+1)^dim instead of @c (2(2r+1))^dim.
         */
        template <std::size_t n_par, std::size_t radius, std::size_t n_comp, class LCA, class Value>
        struct stencil_contraction
        {
            static constexpr std::size_t dim    = LCA::dim;
            static constexpr std::size_t size   = PredictionCoefficients<radius>::size;
            static constexpr std::size_t n_rows = ce_pow(n_par, dim - 1); ///< fine rows predicted together
            /// Coarse cells per chunk, so that the buffers stay within 128 KiB whatever the dimension.
            static constexpr std::size_t chunk  = std::clamp<std::size_t>(8192 / (n_rows * n_comp), size + 3, size + 63) - 2 * radius;
            static constexpr std::size_t stride = (chunk + 2 * radius) * n_comp; ///< between two rows of a buffer

            using value_t = typename LCA::interval_t::value_t;
            using row_t   = std::array<value_t, dim - 1>;

            const LCA& lca; ///< the coarse level
            const Value* src;
            const std::array<std::array<double, size>, 2>& c; ///< coefficients of the low and the high child
            row_t parent;                                     ///< the coarse row
            row_t parity;                                     ///< parity of the first fine row in each direction above x
            value_t x_lo = 0;                                 ///< the coarse columns [x_lo, x_hi) of the current chunk
            value_t x_hi = 0;

            /**
             * Contracts directions @c 1..N of the rows of the stencil found in @c lca[N]
             * between @a start and @a end. Writes @c n_par^N rows into @a out, where the
             * parity of direction @c N varies slowest, and returns it. Direction 0 is the
             * source row itself, which is contiguous over the columns of the chunk.
             */
            template <std::size_t N>
            SAMURAI_INLINE const auto* contract(std::size_t start, std::size_t end, double* out) const
            {
                if constexpr (N == 0)
                {
                    const auto& x_interval = find(0, start, end, x_lo);
                    assert(x_interval.contains(x_hi - 1) && "prediction: a row of the stencil is not contiguous");
                    return src + static_cast<std::size_t>(x_interval.index + x_lo) * n_comp;
                }
                else
                {
                    constexpr std::size_t n_sub = ce_pow(n_par, N - 1);
                    const std::size_t len       = static_cast<std::size_t>(x_hi - x_lo) * n_comp;
                    std::array<double, N == 1 ? 0 : n_sub * stride> sub;

                    for (std::size_t k = 0; k < size; ++k)
                    {
                        const value_t y  = parent[N - 1] + static_cast<value_t>(k) - static_cast<value_t>(radius);
                        const auto off   = static_cast<std::size_t>(find(N, start, end, y).index + y);
                        const auto* rows = contract<N - 1>(lca.offsets(N)[off], lca.offsets(N)[off + 1], sub.data());

                        for (std::size_t q = 0; q < n_par; ++q)
                        {
                            const double ck = c[static_cast<std::size_t>(parity[N - 1]) + q][k];
                            for (std::size_t s = 0; s < n_sub; ++s)
                            {
                                double* o     = out + (q * n_sub + s) * stride;
                                const auto* r = rows + s * stride;
                                for (std::size_t m = 0; m < len; ++m)
                                {
                                    o[m] = (k == 0 ? 0. : o[m]) + ck * r[m];
                                }
                            }
                        }
                    }
                    return static_cast<const double*>(out);
                }
            }

            /// The interval of @c lca[d], between @a start and @a end, that contains @a v.
            SAMURAI_INLINE const auto& find(std::size_t d, std::size_t start, std::size_t end, value_t v) const
            {
                const auto first = lca[d].cbegin();
                const auto pos   = interval_search(first + static_cast<std::ptrdiff_t>(start), first + static_cast<std::ptrdiff_t>(end), v);
                assert(pos >= 0 && "prediction: a cell of the stencil is missing");
                return lca[d][start + static_cast<std::size_t>(pos)];
            }
        };

        /**
         * Predicts the fine cells @c [f_lo, f_hi) of the rows @c first + {0..n_par-1}^(dim-1)
         * of @a fine_level from their parents, one level below.
         */
        template <std::size_t n_par, std::size_t radius, class DEST, class SRC, class Row>
        SAMURAI_INLINE void predict_rows(DEST& dest,
                                         const SRC& src,
                                         std::size_t fine_level,
                                         typename SRC::mesh_t::interval_t::value_t f_lo,
                                         typename SRC::mesh_t::interval_t::value_t f_hi,
                                         const Row& first)
        {
            using mesh_id_t       = typename SRC::mesh_t::mesh_id_t;
            constexpr auto n_comp = SRC::n_comp;
            const auto& lca       = src.mesh()[mesh_id_t::reference][fine_level - 1];
            using contraction_t = stencil_contraction<n_par, radius, n_comp, std::decay_t<decltype(lca)>, std::decay_t<decltype(*src.data())>>;
            using value_t         = typename contraction_t::value_t;
            constexpr auto dim    = contraction_t::dim;
            constexpr auto stride = contraction_t::stride;
            constexpr auto chunk  = static_cast<value_t>(contraction_t::chunk);

            const std::array<std::array<double, contraction_t::size>, 2> c = {
                {prediction_coefficients<radius>(0, 0).c, prediction_coefficients<radius>(1, 0).c}
            };

            contraction_t contraction{lca, src.data(), c, {}, {}};
            for (std::size_t d = 0; d < dim - 1; ++d)
            {
                contraction.parent[d] = first[d] >> 1;
                contraction.parity[d] = first[d] & 1;
            }

            // Memory offset of the column f_lo in each fine row, in the order of the contraction.
            std::array<std::size_t, contraction_t::n_rows> dest_rows;
            for (std::size_t q = 0; q < dest_rows.size(); ++q)
            {
                typename contraction_t::row_t fine_row;
                for (std::size_t d = 0, digits = q; d < dim - 1; ++d, digits /= n_par)
                {
                    fine_row[d] = first[d] + static_cast<value_t>(digits % n_par);
                }
                dest_rows[q] = memory_offset(dest.mesh(), {fine_level, f_lo, fine_row});
            }

            std::array<double, contraction_t::n_rows * stride> rows;
            const value_t c_lo = f_lo >> 1;
            const value_t c_hi = ((f_hi - 1) >> 1) + 1;
            for (value_t a = c_lo; a < c_hi; a += chunk)
            {
                const value_t b   = std::min(c_hi, a + chunk);
                contraction.x_lo  = a - static_cast<value_t>(radius);
                contraction.x_hi  = b + static_cast<value_t>(radius);
                const auto* above = contraction.template contract<dim - 1>(0, lca[dim - 1].size(), rows.data());

                // x comes last: the fine cell f reads the 2r+1 columns centred on its parent.
                for (std::size_t q = 0; q < dest_rows.size(); ++q)
                {
                    for (value_t f = std::max(f_lo, 2 * a); f < std::min(f_hi, 2 * b); ++f)
                    {
                        const auto& cf = c[static_cast<std::size_t>(f & 1)];
                        const auto* t  = above + q * stride + static_cast<std::size_t>((f >> 1) - a) * n_comp;
                        auto* d        = dest.data() + (dest_rows[q] + static_cast<std::size_t>(f - f_lo)) * n_comp;
                        for (std::size_t n = 0; n < n_comp; ++n)
                        {
                            double v = 0.;
                            for (std::size_t s = 0; s < contraction_t::size; ++s)
                            {
                                v += cf[s] * t[s * n_comp + n];
                            }
                            d[n] = v;
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

        /**
         * With @a dest_on_level, predicts the @c 2^dim children of every cell of the interval
         * @c i at @c level + 1; otherwise, the cells of @c i at @c level from their parents.
         */
        template <class DEST, class SRC, std::size_t radius, bool dest_on_level>
        SAMURAI_INLINE void operator()(Dim<dim>,
                                       DEST& dest,
                                       const SRC& src,
                                       std::integral_constant<std::size_t, radius>,
                                       std::integral_constant<bool, dest_on_level>) const
        {
            static_assert(DEST::n_comp == SRC::n_comp, "Source and destination fields must have the same number of components");
            using value_t = typename TInterval::value_t;

            constexpr std::size_t n_par = dest_on_level ? 2 : 1; // fine rows per row of i, in each direction
            const auto scale            = static_cast<value_t>(n_par);

            std::array<value_t, dim - 1> first; // the first fine row
            for (std::size_t d = 0; d < dim - 1; ++d)
            {
                first[d] = scale * index[d];
            }
            detail::predict_rows<n_par, radius>(dest, src, level + n_par - 1, scale * i.start, scale * i.end, first);
        }
    };

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
