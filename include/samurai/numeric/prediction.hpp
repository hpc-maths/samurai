// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <xtensor/views/xview.hpp>

#include "../field/concepts.hpp"
#include "../operators_base.hpp"
#include "../prediction_shifts.hpp"
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
        static_assert(DEST::n_comp == SRC::n_comp, "Source and destination fields must have the same number of components");

        constexpr std::size_t order = 2 * pred_stencil_size + 1;
        using value_t               = typename TInterval::value_t;

        for_each_prediction_shift_run<pred_stencil_size>(
            prediction_domain(src.mesh(), level),
            i,
            index,
            [&](const auto& run, const auto& shifts)
            {
                // (even index coefficients, odd index coefficients), per direction: near a boundary
                // the stencil is shifted inward, by a different amount in each direction.
                std::array<std::array<std::array<double, order>, 2>, dim> interp_coeff_pair;
                xt::xtensor_fixed<value_t, xt::xshape<dim>> stencil_start;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const auto& even        = prediction_coefficients<pred_stencil_size>(0, shift_of(shifts, d));
                    const auto& odd         = prediction_coefficients<pred_stencil_size>(1, shift_of(shifts, d));
                    interp_coeff_pair[d][0] = even.c;
                    interp_coeff_pair[d][1] = odd.c;
                    stencil_start[d]        = static_cast<value_t>(even.start);
                }

                // Compute the memory accessors for the source data
                // For example, in 2D, for a prediction stencil of size 1, we need to access the following cells in the source field
                //
                // (level, i-1, j+1) (level, i, j+1) (level, i+1, j+1)
                // (level, i-1, j  ) (level, i, j  ) (level, i+1, j  )
                // (level, i-1, j-1) (level, i, j-1) (level, i+1, j-1)
                //
                // Since the data are contiguous in the i direction, we just have to compute the memory addresses of the first column.

                std::array<std::size_t, ce_pow(order, dim)> src_offsets;
                std::size_t ind = 0;
                static_nested_loop<dim, 0, order>(
                    [&](const auto& stencil)
                    {
                        auto new_index     = index + xt::view(stencil, xt::range(1, dim)) + xt::view(stencil_start, xt::range(1, dim));
                        src_offsets[ind++] = memory_offset(src.mesh(), {level, run.start + stencil[0] + stencil_start[0], new_index});
                    });

                // Compute the memory accessors for the destination data
                // For example, in 2D, we need to access the following cells in the destination field
                //
                // (level + 1, 2i  , 2j  ) (level + 1, 2i+1, 2j  )
                // (level + 1, 2i  , 2j+1) (level + 1, 2i+1, 2j+1)
                //
                // Since the data are contiguous in the i direction, once again, we just have to compute the memory addresses of the first
                // column.

                std::array<std::size_t, 1ULL << dim> dest_offsets;
                ind = 0;
                static_nested_loop<dim - 1, 0, 2>(
                    [&](const auto& stencil)
                    {
                        auto new_index        = 2 * index + stencil;
                        dest_offsets[ind]     = memory_offset(dest.mesh(), {level + 1, 2 * run.start, new_index});
                        dest_offsets[ind + 1] = dest_offsets[ind] + 1;
                        ind += 2;
                    });

                const auto* src_data = src.data();
                auto* dest_data      = dest.data();

                std::array<double, (1ULL << dim) * SRC::n_comp> dest_values{};
                for (std::size_t i_c = 0, i_f = 0; i_c < run.size(); ++i_c, i_f += 2)
                {
                    dest_values.fill(0);
                    std::size_t io = 0;
                    static_nested_loop<dim, 0, order>(
                        [&](const auto& stencil)
                        {
                            std::array<double, SRC::n_comp> field_ijk{};
                            for (std::size_t n = 0; n < SRC::n_comp; ++n)
                            {
                                field_ijk[n] = src_data[(src_offsets[io] + i_c) * SRC::n_comp + n];
                            }
                            ++io;

                            std::size_t ind = 0;
                            std::apply(
                                [&](const auto&... s)
                                {
                                    for (std::size_t n = 0; n < SRC::n_comp; ++n)
                                    {
                                        (void)std::initializer_list<int>{
                                            ((dest_values[ind++] += field_ijk[n]
                                                                  * std::apply(
                                                                        [&](const auto&... ki)
                                                                        {
                                                                            std::size_t is = 0;
                                                                            double coeff   = 1.;
                                                                            ((coeff *= interp_coeff_pair[is][ki][static_cast<std::size_t>(
                                                                                  stencil[is])],
                                                                              ++is),
                                                                             ...);
                                                                            return coeff;
                                                                        },
                                                                        s)),
                                             0)...};
                                    }
                                },
                                make_index_ranges<dim, 0, 2>());
                        });

                    std::size_t id = 0;
                    std::apply(
                        [&](const auto&... s)
                        {
                            for (std::size_t n = 0; n < SRC::n_comp; ++n)
                            {
                                ((dest_data[(s + i_f) * SRC::n_comp + n] = dest_values[id++]), ...);
                            }
                        },
                        dest_offsets);
                }
            });
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
        static_assert(DEST::n_comp == SRC::n_comp, "Source and destination fields must have the same number of components");

        constexpr std::size_t order = 2 * pred_stencil_size + 1;
        using value_t               = typename TInterval::value_t;

        const auto* src_data = src.data();
        auto* dest_data      = dest.data();

        auto dest_offset = memory_offset(dest.mesh(), {level, i.start, index});

        // Here the destination is the fine level and the stencil lives one level up, so the
        // shift is classified there: on the parents of `i`, and over the coarse domain.
        const TInterval parents{i.start >> 1, ((i.end - 1) >> 1) + 1, i.index};

        // Built cell by cell rather than as `index >> 1`: that is a lazy xtensor expression,
        // and storing one keeps a reference to the scalar it was written against.
        xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> parent_index;
        for (std::size_t d = 0; d + 1 < dim; ++d)
        {
            parent_index[d] = index[d] >> 1;
        }

        for_each_prediction_shift_run<pred_stencil_size>(
            prediction_domain(src.mesh(), level - 1),
            parents,
            parent_index,
            [&](const auto& run, const auto& shifts)
            {
                // The fine cells of `i` whose parent this run holds.
                const value_t fine_start = std::max(i.start, 2 * run.start);
                const value_t fine_end   = std::min(i.end, 2 * run.end);
                if (fine_start >= fine_end)
                {
                    return;
                }

                // (even index coefficients, odd index coefficients), per direction
                std::array<std::array<std::array<double, order>, 2>, dim> interp_coeff_pair;
                xt::xtensor_fixed<value_t, xt::xshape<dim>> stencil_start;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const auto& even        = prediction_coefficients<pred_stencil_size>(0, shift_of(shifts, d));
                    const auto& odd         = prediction_coefficients<pred_stencil_size>(1, shift_of(shifts, d));
                    interp_coeff_pair[d][0] = even.c;
                    interp_coeff_pair[d][1] = odd.c;
                    stencil_start[d]        = static_cast<value_t>(even.start);
                }

                std::array<std::size_t, dim> parity;
                parity[0] = (fine_start & 1) ? 1 : 0;
                for (std::size_t d = 1; d < dim; ++d)
                {
                    parity[d] = (index[d - 1] & 1) ? 1 : 0;
                }

                std::array<std::size_t, ce_pow(order, dim)> src_offsets;
                std::size_t ind = 0;
                static_nested_loop<dim, 0, order>(
                    [&](const auto& stencil)
                    {
                        auto new_index = parent_index + xt::view(stencil, xt::range(1, dim)) + xt::view(stencil_start, xt::range(1, dim));
                        src_offsets[ind++] = memory_offset(src.mesh(), {level - 1, run.start + stencil[0] + stencil_start[0], new_index});
                    });

                auto apply_pred = [&](const auto& i_f, const auto& i_c)
                {
                    std::array<double, SRC::n_comp> dest_value{};
                    std::size_t io = 0;
                    static_nested_loop<dim, 0, order>(
                        [&](const auto& stencil)
                        {
                            for (std::size_t n = 0; n < SRC::n_comp; ++n)
                            {
                                auto field_ijk = src_data[(src_offsets[io] + i_c) * SRC::n_comp + n];

                                dest_value[n] += field_ijk
                                               * std::apply(
                                                     [&](const auto&... ki)
                                                     {
                                                         std::size_t is = 0;
                                                         double coeff   = 1.;
                                                         ((coeff *= interp_coeff_pair[is][ki][static_cast<std::size_t>(stencil[is])], ++is),
                                                          ...);
                                                         return coeff;
                                                     },
                                                     parity);
                            }
                            io++;
                        });

                    for (std::size_t n = 0; n < SRC::n_comp; ++n)
                    {
                        dest_data[(dest_offset + i_f) * SRC::n_comp + n] = dest_value[n];
                    }
                    parity[0] = (parity[0] & 1) ? 0 : 1;
                };

                for (value_t x = fine_start; x < fine_end; ++x)
                {
                    apply_pred(static_cast<std::size_t>(x - i.start), static_cast<std::size_t>((x >> 1) - run.start));
                }
            });
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
