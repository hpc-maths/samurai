// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <type_traits>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>

#include "../operators_base.hpp"
#include "../storage/utils.hpp"
#ifdef SAMURAI_CHECK_NAN
#include "../io/hdf5.hpp"
#endif

#ifdef SAMURAI_CHECK_NAN
#include <mpi.h>
#endif

namespace samurai
{
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

        template <class T1, class T2>
        void operator()(Dim<dim>, T1& dest, const T2& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, true>) const;

        template <class T1, class T2>
        void operator()(Dim<dim>, T1& dest, const T2& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, false>) const;

        template <class T1, class T2, std::size_t order>
        void
        operator()(Dim<dim>, T1& dest, const T2& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, true>) const;

        template <class T1, class T2, std::size_t order>
        void
        operator()(Dim<dim>, T1& dest, const T2& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, false>) const;
    };

    template <std::size_t dim, class TInterval>
    template <class DEST, class SRC>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<dim>,
                                                                  DEST& dest,
                                                                  const SRC& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, true>) const
    {
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
                for (std::size_t n = 0; n < SRC::n_comp; ++n)
                {
                    dest_data[dest_offsets[s] + i_f * SRC::n_comp + n]       = src_data[src_offset + i_c * SRC::n_comp + n];
                    dest_data[dest_offsets[s] + (i_f + 1) * SRC::n_comp + n] = src_data[src_offset + i_c * SRC::n_comp + n];
                }
            }
        }
    }

    template <std::size_t dim, std::size_t b, std::size_t e>
    consteval auto make_index_ranges()
    {
        constexpr std::size_t base = e - b;
        static_assert(base > 0, "make_index_ranges requires e > b");

        constexpr std::size_t count = ce_pow(base, dim);
        std::array<std::array<std::size_t, dim>, count> result{};

        for (std::size_t n = 0; n < count; ++n)
        {
            std::size_t value = n;
            for (std::size_t d = 0; d < dim; ++d)
            {
                result[n][d] = b + (value % base);
                value /= base;
            }
        }
        return result;
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

        // (even index coefficients, odd index coefficients)
        std::array<std::array<double, order>, 2> interp_coeff_pair = {
            {interp_coeffs<order>(1.), interp_coeffs<order>(-1.)}
        };

        // Compute the memory accessors for the source data
        // For example, in 2D, for a prediction stencil of size 1, we need to access the following cells in the source field
        //
        // (level, i-1, j+1) (level, i, j+1) (level, i+1, j+1)
        // (level, i-1, j  ) (level, i, j  ) (level, i+1, j  )
        // (level, i-1, j-1) (level, i, j-1) (level, i+1, j-1)
        //
        // Since the data are contiguous in the i direction, we just have to compute the memory adresses of the first column.

        std::array<std::size_t, ce_pow(order, dim)> src_offsets;
        std::size_t ind = 0;
        static_nested_loop<dim, -static_cast<int>(pred_stencil_size), static_cast<int>(pred_stencil_size) + 1>(
            [&](const auto& stencil)
            {
                auto new_index     = index + xt::view(stencil, xt::range(1, dim));
                src_offsets[ind++] = memory_offset(src.mesh(), {level, i.start + stencil[0], new_index});
            });

        // Compute the memory accessors for the destination data
        // For example, in 2D, we need to access the following cells in the destination field
        //
        // (level + 1, 2i  , 2j  ) (level + 1, 2i+1, 2j  )
        // (level + 1, 2i+1, 2j+1) (level + 1, 2i+1, 2j+1)
        //
        // Since the data are contiguous in the i direction, once again, we just have to compute the memory adresses of the first column.

        std::array<std::size_t, 1ULL << dim> dest_offsets;
        ind = 0;
        static_nested_loop<dim - 1, 0, 2>(
            [&](const auto& stencil)
            {
                auto new_index        = 2 * index + stencil;
                dest_offsets[ind]     = memory_offset(dest.mesh(), {level + 1, 2 * i.start, new_index});
                dest_offsets[ind + 1] = dest_offsets[ind] + 1;
                ind += 2;
            });

        const auto* src_data = src.data();
        auto* dest_data      = dest.data();

        std::array<double, (1ULL << dim) * SRC::n_comp> dest_values{};
        for (std::size_t i_c = 0, i_f = 0; i_c < i.size(); ++i_c, i_f += 2)
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
                                                                    ((coeff *= interp_coeff_pair[ki][static_cast<std::size_t>(stencil[is])],
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
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<dim>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, false>) const
    {
        const auto* src_data = src.data();
        auto* dest_data      = dest.data();

        auto even_i = i.even_elements();
        if (even_i.is_valid())
        {
            auto src_offset  = memory_offset(src.mesh(), {level - 1, even_i.start >> 1, index >> 1});
            auto dest_offset = memory_offset(dest.mesh(), {level, even_i.start, index});
            for (std::size_t i_f = 0, i_c = 0; i_f < even_i.size(); i_f += 2, ++i_c)
            {
                dest_data[dest_offset + i_f] = src_data[src_offset + i_c];
            }
        }

        auto odd_i = i.odd_elements();
        if (odd_i.is_valid())
        {
            auto src_offset  = memory_offset(src.mesh(), {level - 1, odd_i.start >> 1, index >> 1});
            auto dest_offset = memory_offset(dest.mesh(), {level, odd_i.start, index});
            for (std::size_t i_f = 0, i_c = 0; i_f < odd_i.size(); i_f += 2, ++i_c)
            {
                dest_data[dest_offset + i_f] = src_data[src_offset + i_c];
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
        constexpr std::size_t order = 2 * pred_stencil_size + 1;
        using value_t               = typename TInterval::value_t;

        // (even index coefficients, odd index coefficients)
        std::array<std::array<double, order>, 2> interp_coeff_pair = {
            {interp_coeffs<order>(1.), interp_coeffs<order>(-1.)}
        };

        const auto* src_data = src.data();
        auto* dest_data      = dest.data();

        auto dest_offset = memory_offset(dest.mesh(), {level, i.start, index});

        std::array<std::size_t, dim> parity;
        parity[0] = (i.start & 1) ? 1 : 0;
        for (std::size_t d = 1; d < dim; ++d)
        {
            parity[d] = (index[d - 1] & 1) ? 1 : 0;
        }

        std::array<std::size_t, ce_pow(order, dim)> src_offsets;
        std::size_t ind = 0;
        static_nested_loop<dim, -static_cast<int>(pred_stencil_size), static_cast<int>(pred_stencil_size) + 1>(
            [&](const auto& stencil)
            {
                auto new_index     = (index >> 1) + xt::view(stencil, xt::range(1, dim));
                src_offsets[ind++] = memory_offset(src.mesh(), {level - 1, (i.start >> 1) + stencil[0], new_index});
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
                                                 ((coeff *= interp_coeff_pair[ki][static_cast<std::size_t>(stencil[is])], ++is), ...);
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

        for (std::size_t i_f = 0; i_f < i.size(); ++i_f)
        {
            apply_pred(i_f, static_cast<std::size_t>(((i.start + static_cast<value_t>(i_f)) >> 1) - (i.start >> 1)));
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

    template <std::size_t order, bool dest_on_level, class T1, class T2>
    SAMURAI_INLINE auto prediction(T1& field_dest, const T2& field_src)
    {
        return make_field_operator_function<prediction_op>(field_dest,
                                                           field_src,
                                                           std::integral_constant<std::size_t, order>{},
                                                           std::integral_constant<bool, dest_on_level>{});
    }
}
