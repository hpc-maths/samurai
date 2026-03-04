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
        void operator()(Dim<1>, T1& dest, const T2& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, true>) const;

        template <class T1, class T2>
        void operator()(Dim<1>, T1& dest, const T2& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, false>) const;

        template <class T1, class T2, std::size_t order>
        void
        operator()(Dim<1>, T1& dest, const T2& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, true>) const;

        template <class T1, class T2, std::size_t order>
        void
        operator()(Dim<1>, T1& dest, const T2& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, false>) const;

        template <class T1, class T2>
        void operator()(Dim<2>, T1& dest, const T2& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, true>) const;

        template <class T1, class T2>
        void operator()(Dim<2>, T1& dest, const T2& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, false>) const;

        template <class T1, class T2, std::size_t order>
        void
        operator()(Dim<2>, T1& dest, const T2& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, true>) const;

        template <class T1, class T2, std::size_t order>
        void
        operator()(Dim<2>, T1& dest, const T2& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, false>) const;

        template <class T1, class T2>
        void operator()(Dim<3>, T1& dest, const T2& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, true>) const;

        template <class T1, class T2>
        void operator()(Dim<3>, T1& dest, const T2& src, std::integral_constant<std::size_t, 0>, std::integral_constant<bool, false>) const;

        template <class T1, class T2, std::size_t order>
        void
        operator()(Dim<3>, T1& dest, const T2& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, true>) const;

        template <class T1, class T2, std::size_t order>
        void
        operator()(Dim<3>, T1& dest, const T2& src, std::integral_constant<std::size_t, order>, std::integral_constant<bool, false>) const;
    };

    template <std::size_t dim, class TInterval>
    template <class T1, class T2>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<1>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, true>) const
    {
        auto ii = i << 1;
        ii.step = 2;

        dest(level + 1, ii)     = src(level, i);
        dest(level + 1, ii + 1) = src(level, i);
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<1>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, false>) const
    {
        auto even_i = i.even_elements();
        if (even_i.is_valid())
        {
            auto coarse_even_i  = even_i >> 1;
            dest(level, even_i) = src(level - 1, coarse_even_i);
        }

        auto odd_i = i.odd_elements();
        if (odd_i.is_valid())
        {
            auto coarse_odd_i  = odd_i >> 1;
            dest(level, odd_i) = src(level - 1, coarse_odd_i);
        }
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2, std::size_t order>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<1>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, order>,
                                                                  std::integral_constant<bool, true>) const
    {
        auto ii = i << 1;
        ii.step = 2;

        using value_t    = typename TInterval::value_t;
        auto sorder      = static_cast<value_t>(order);
        auto interp_even = interp_coeffs<2 * order + 1>(1.);
        auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

        dest(level + 1, ii)     = 0;
        dest(level + 1, ii + 1) = 0;

        for (value_t ki = 0; ki < 2 * sorder + 1; ++ki)
        {
            std::size_t uki = static_cast<std::size_t>(ki);
            auto field_ik   = src(level, i + ki - sorder);
            dest(level + 1, ii) += interp_even[uki] * field_ik;
            dest(level + 1, ii + 1) += interp_odd[uki] * field_ik;
        }
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2, std::size_t order>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<1>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, order>,
                                                                  std::integral_constant<bool, false>) const
    {
        using value_t    = typename TInterval::value_t;
        auto sorder      = static_cast<value_t>(order);
        auto interp_even = interp_coeffs<2 * order + 1>(1.);
        auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

        auto apply_pred = [&](const auto& i_f, const auto& i_c, const auto& interpi)
        {
            dest(level, i_f) = 0;

            for (value_t ki = 0; ki < 2 * sorder + 1; ++ki)
            {
                std::size_t uki = static_cast<std::size_t>(ki);
                auto field_ij   = src(level - 1, i_c + ki - sorder);
                dest(level, i_f) += interpi[uki] * field_ij;
            };
        };

        auto even_i = i.even_elements();
        if (even_i.is_valid())
        {
            apply_pred(even_i, even_i >> 1, interp_even);
        }

        auto odd_i = i.odd_elements();
        if (odd_i.is_valid())
        {
            apply_pred(odd_i, odd_i >> 1, interp_odd);
        }
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<2>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, true>) const
    {
        auto ii = i << 1;
        ii.step = 2;

        auto jj = j << 1;

        dest(level + 1, ii, jj)         = src(level, i, j);
        dest(level + 1, ii + 1, jj)     = src(level, i, j);
        dest(level + 1, ii, jj + 1)     = src(level, i, j);
        dest(level + 1, ii + 1, jj + 1) = src(level, i, j);
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<2>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, false>) const
    {
        if (j & 1)
        {
            auto even_i = i.even_elements();
            if (even_i.is_valid())
            {
                auto coarse_even_i     = even_i >> 1;
                dest(level, even_i, j) = src(level - 1, coarse_even_i, j >> 1);
            }

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
            {
                auto coarse_odd_i     = odd_i >> 1;
                dest(level, odd_i, j) = src(level - 1, coarse_odd_i, j >> 1);
            }
        }
        else
        {
            auto even_i = i.even_elements();
            if (even_i.is_valid())
            {
                auto coarse_even_i     = even_i >> 1;
                dest(level, even_i, j) = src(level - 1, coarse_even_i, j >> 1);
            }

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
            {
                auto coarse_odd_i     = odd_i >> 1;
                dest(level, odd_i, j) = src(level - 1, coarse_odd_i, j >> 1);
            }
        }
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2, std::size_t order>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<2>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, order>,
                                                                  std::integral_constant<bool, true>) const
    {
        auto ii = i << 1;
        ii.step = 2;

        auto jj = j << 1;

        using value_t    = typename TInterval::value_t;
        auto sorder      = static_cast<value_t>(order);
        auto interp_even = interp_coeffs<2 * order + 1>(1.);
        auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

        dest(level + 1, ii, jj)         = 0;
        dest(level + 1, ii + 1, jj)     = 0;
        dest(level + 1, ii, jj + 1)     = 0;
        dest(level + 1, ii + 1, jj + 1) = 0;

        for (value_t kj = 0; kj < 2 * sorder + 1; ++kj)
        {
            std::size_t ukj = static_cast<std::size_t>(kj);
            for (value_t ki = 0; ki < 2 * sorder + 1; ++ki)
            {
                std::size_t uki = static_cast<std::size_t>(ki);
                auto field_ij   = src(level, i + ki - sorder, j + kj - sorder);
                dest(level + 1, ii, jj) += interp_even[uki] * interp_even[ukj] * field_ij;
                dest(level + 1, ii + 1, jj) += interp_odd[uki] * interp_even[ukj] * field_ij;
                dest(level + 1, ii, jj + 1) += interp_even[uki] * interp_odd[ukj] * field_ij;
                dest(level + 1, ii + 1, jj + 1) += interp_odd[uki] * interp_odd[ukj] * field_ij;
            }
        }
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2, std::size_t order>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<2>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, order>,
                                                                  std::integral_constant<bool, false>) const
    {
        using value_t    = typename TInterval::value_t;
        auto sorder      = static_cast<value_t>(order);
        auto interp_even = interp_coeffs<2 * order + 1>(1.);
        auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

        auto apply_pred = [&](const auto& i_f, const auto& i_c, const auto& interpi, const auto& interpj)
        {
            dest(level, i_f, j) = 0;

            for (value_t kj = 0; kj < 2 * sorder + 1; ++kj)
            {
                std::size_t ukj = static_cast<std::size_t>(kj);
                for (value_t ki = 0; ki < 2 * sorder + 1; ++ki)
                {
                    std::size_t uki = static_cast<std::size_t>(ki);
                    auto field_ij   = src(level - 1, i_c + ki - sorder, (j >> 1) + kj - sorder);
#ifdef SAMURAI_CHECK_NAN
                    bool is_nan = false;
                    if constexpr (T2::is_scalar)
                    {
                        if (std::isnan(field_ij))
                        {
                            is_nan = true;
                        }
                    }
                    else
                    {
                        if (xt::any(xt::isnan(field_ij)))
                        {
                            is_nan = true;
                        }
                    }
                    if (is_nan)
                    {
                        std::cerr << "NaN detected in prediction_op at level " << level - 1 << ", i " << i_c + ki - sorder << ", j "
                                  << (j >> 1) + kj - sorder << std::endl;
                        exit(1);
                    }
#endif
                    dest(level, i_f, j) += interpi[uki] * interpj[ukj] * field_ij;
                };
            }
        };

        if (j & 1)
        {
            auto even_i = i.even_elements();
            if (even_i.is_valid())
            {
                apply_pred(even_i, even_i >> 1, interp_even, interp_odd);
            }

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
            {
                apply_pred(odd_i, odd_i >> 1, interp_odd, interp_odd);
            }
        }
        else
        {
            auto even_i = i.even_elements();
            if (even_i.is_valid())
            {
                apply_pred(even_i, even_i >> 1, interp_even, interp_even);
            }

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
            {
                apply_pred(odd_i, odd_i >> 1, interp_odd, interp_even);
            }
        }
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<3>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, true>) const
    {
        auto ii = i << 1;
        ii.step = 2;

        auto jj = j << 1;
        auto kk = k << 1;

        dest(level + 1, ii, jj, kk)             = src(level, i, j, k);
        dest(level + 1, ii + 1, jj, kk)         = src(level, i, j, k);
        dest(level + 1, ii, jj + 1, kk)         = src(level, i, j, k);
        dest(level + 1, ii + 1, jj + 1, kk)     = src(level, i, j, k);
        dest(level + 1, ii, jj, kk + 1)         = src(level, i, j, k);
        dest(level + 1, ii + 1, jj, kk + 1)     = src(level, i, j, k);
        dest(level + 1, ii, jj + 1, kk + 1)     = src(level, i, j, k);
        dest(level + 1, ii + 1, jj + 1, kk + 1) = src(level, i, j, k);
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<3>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, 0>,
                                                                  std::integral_constant<bool, false>) const
    {
        auto even_i = i.even_elements();
        if (even_i.is_valid())
        {
            auto coarse_even_i        = even_i >> 1;
            dest(level, even_i, j, k) = src(level - 1, coarse_even_i, j >> 1, k >> 1);
        }

        auto odd_i = i.odd_elements();
        if (odd_i.is_valid())
        {
            auto coarse_odd_i        = odd_i >> 1;
            dest(level, odd_i, j, k) = src(level - 1, coarse_odd_i, j >> 1, k >> 1);
        }
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2, std::size_t order>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<3>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, order>,
                                                                  std::integral_constant<bool, true>) const
    {
        auto i_f = i << 1;
        i_f.step = 2;

        auto j_f = j << 1;
        auto k_f = k << 1;

        using value_t    = typename TInterval::value_t;
        auto sorder      = static_cast<value_t>(order);
        auto interp_even = interp_coeffs<2 * order + 1>(1.);
        auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

        dest(level + 1, i_f, j_f, k_f)             = 0;
        dest(level + 1, i_f + 1, j_f, k_f)         = 0;
        dest(level + 1, i_f, j_f + 1, k_f)         = 0;
        dest(level + 1, i_f + 1, j_f + 1, k_f)     = 0;
        dest(level + 1, i_f, j_f, k_f + 1)         = 0;
        dest(level + 1, i_f + 1, j_f, k_f + 1)     = 0;
        dest(level + 1, i_f, j_f + 1, k_f + 1)     = 0;
        dest(level + 1, i_f + 1, j_f + 1, k_f + 1) = 0;

        for (value_t kk = 0; kk < 2 * sorder + 1; ++kk)
        {
            std::size_t ukk = static_cast<std::size_t>(kk);
            for (value_t kj = 0; kj < 2 * sorder + 1; ++kj)
            {
                std::size_t ukj = static_cast<std::size_t>(kj);
                for (value_t ki = 0; ki < 2 * sorder + 1; ++ki)
                {
                    std::size_t uki = static_cast<std::size_t>(ki);
                    auto field_ijk  = src(level, i + ki - sorder, j + kj - sorder, k + kk - sorder);
                    dest(level + 1, i_f, j_f, k_f) += interp_even[uki] * interp_even[ukj] * interp_even[ukk] * field_ijk;
                    dest(level + 1, i_f + 1, j_f, k_f) += interp_odd[uki] * interp_even[ukj] * interp_even[ukk] * field_ijk;
                    dest(level + 1, i_f, j_f + 1, k_f) += interp_even[uki] * interp_odd[ukj] * interp_even[ukk] * field_ijk;
                    dest(level + 1, i_f + 1, j_f + 1, k_f) += interp_odd[uki] * interp_odd[ukj] * interp_even[ukk] * field_ijk;
                    dest(level + 1, i_f, j_f, k_f + 1) += interp_even[uki] * interp_even[ukj] * interp_odd[ukk] * field_ijk;
                    dest(level + 1, i_f + 1, j_f, k_f + 1) += interp_odd[uki] * interp_even[ukj] * interp_odd[ukk] * field_ijk;
                    dest(level + 1, i_f, j_f + 1, k_f + 1) += interp_even[uki] * interp_odd[ukj] * interp_odd[ukk] * field_ijk;
                    dest(level + 1, i_f + 1, j_f + 1, k_f + 1) += interp_odd[uki] * interp_odd[ukj] * interp_odd[ukk] * field_ijk;
                }
            }
        }
    }

    template <std::size_t dim, class TInterval>
    template <class T1, class T2, std::size_t order>
    SAMURAI_INLINE void prediction_op<dim, TInterval>::operator()(Dim<3>,
                                                                  T1& dest,
                                                                  const T2& src,
                                                                  std::integral_constant<std::size_t, order>,
                                                                  std::integral_constant<bool, false>) const
    {
        using value_t    = typename TInterval::value_t;
        auto sorder      = static_cast<value_t>(order);
        auto interp_even = interp_coeffs<2 * order + 1>(1.);
        auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

        auto apply_pred = [&](const auto& i_f, const auto& i_c, const auto& interpi, const auto interpj, const auto interpk)
        {
            dest(level, i_f, j, k) = 0;

            for (value_t kk = 0; kk < 2 * sorder + 1; ++kk)
            {
                std::size_t ukk = static_cast<std::size_t>(kk);
                for (value_t kj = 0; kj < 2 * sorder + 1; ++kj)
                {
                    std::size_t ukj = static_cast<std::size_t>(kj);
                    for (value_t ki = 0; ki < 2 * sorder + 1; ++ki)
                    {
                        std::size_t uki = static_cast<std::size_t>(ki);
                        auto field_ijk  = src(level - 1, i_c + ki - sorder, (j >> 1) + kj - sorder, (k >> 1) + kk - sorder);
#ifdef SAMURAI_CHECK_NAN
                        bool is_nan = false;
                        if constexpr (T2::is_scalar)
                        {
                            if (std::isnan(field_ijk))
                            {
                                is_nan = true;
                            }
                        }
                        else
                        {
                            if (xt::any(xt::isnan(field_ijk)))
                            {
                                is_nan = true;
                            }
                        }
                        if (is_nan)
                        {
                            std::cerr << "NaN detected in prediction_op at level " << level - 1 << ", i " << i_c + ki - sorder << ", j "
                                      << (j >> 1) + kj - sorder << ", k " << (k >> 1) + kk - sorder << std::endl;
                            exit(1);
                        }
#endif
                        dest(level, i_f, j, k) += interpi[uki] * interpj[ukj] * interpk[ukk] * field_ijk;
                    };
                }
            }
        };

        if (k & 1)
        {
            if (j & 1)
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    apply_pred(even_i, even_i >> 1, interp_even, interp_odd, interp_odd);
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    apply_pred(odd_i, odd_i >> 1, interp_odd, interp_odd, interp_odd);
                }
            }
            else
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    apply_pred(even_i, even_i >> 1, interp_even, interp_even, interp_odd);
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    apply_pred(odd_i, odd_i >> 1, interp_odd, interp_even, interp_odd);
                }
            }
        }
        else
        {
            if (j & 1)
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    apply_pred(even_i, even_i >> 1, interp_even, interp_odd, interp_even);
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    apply_pred(odd_i, odd_i >> 1, interp_odd, interp_odd, interp_even);
                }
            }
            else
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    apply_pred(even_i, even_i >> 1, interp_even, interp_even, interp_even);
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    apply_pred(odd_i, odd_i >> 1, interp_odd, interp_even, interp_even);
                }
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
        SAMURAI_INLINE void operator()(Dim<1>,
                                       std::integral_constant<std::size_t, order> o,
                                       std::integral_constant<bool, dest_on_level> dest,
                                       Head& source,
                                       Tail&... sources) const
        {
            prediction_op<dim, interval_t>(level, i)(Dim<1>{}, source, source, o, dest);
            this->operator()(Dim<1>{}, o, dest, sources...);
        }

        template <std::size_t order, bool dest_on_level, class Head, class... Tail>
        SAMURAI_INLINE void operator()(Dim<2>,
                                       std::integral_constant<std::size_t, order> o,
                                       std::integral_constant<bool, dest_on_level> dest,
                                       Head& source,
                                       Tail&... sources) const
        {
            prediction_op<dim, interval_t>(level, i, j)(Dim<2>{}, source, source, o, dest);
            this->operator()(Dim<2>{}, o, dest, sources...);
        }

        template <std::size_t order, bool dest_on_level, class Head, class... Tail>
        SAMURAI_INLINE void operator()(Dim<3>,
                                       std::integral_constant<std::size_t, order> o,
                                       std::integral_constant<bool, dest_on_level> dest,
                                       Head& source,
                                       Tail&... sources) const
        {
            prediction_op<dim, interval_t>(level, i, j, k)(Dim<3>{}, source, source, o, dest);
            this->operator()(Dim<3>{}, o, dest, sources...);
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
