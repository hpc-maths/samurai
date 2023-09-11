// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>
#include <type_traits>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include "../operators_base.hpp"

namespace samurai
{
    template <std::size_t s>
    inline std::array<double, s> interp_coeffs(double sign);

    template <>
    inline std::array<double, 1> interp_coeffs(double)
    {
        return {1};
    }

    template <>
    inline std::array<double, 3> interp_coeffs(double sign)
    {
        return {sign / 8., 1, -sign / 8.};
    }

    template <>
    inline std::array<double, 5> interp_coeffs(double sign)
    {
        return {-sign * 3. / 128., sign * 22. / 128., 1, -sign * 22 / 128., sign * 3. / 128.};
    }

    template <>
    inline std::array<double, 7> interp_coeffs(double sign)
    {
        return {sign * 5. / 1024., -sign * 11. / 256., sign * 201. / 1024., 1, -sign * 201. / 1024., sign * 11. / 256., -sign * 5. / 1024.};
    }

    template <>
    inline std::array<double, 9> interp_coeffs(double sign)
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
    inline std::array<double, 11> interp_coeffs(double sign)
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

    template <std::size_t s>
    inline std::array<double, s> prediction_coeffs();

    template <>
    inline std::array<double, 1> prediction_coeffs<1>()
    {
        return {-1. / 8.};
    }

    template <>
    inline std::array<double, 2> prediction_coeffs<2>()
    {
        return {-22. / 128., 3. / 128.};
    }

    template <>
    inline std::array<double, 3> prediction_coeffs<3>()
    {
        return {-201. / 1024., 11. / 256., -5. / 1024.};
    }

    template <>
    inline std::array<double, 4> prediction_coeffs<4>()
    {
        return {-3461. / 16384., 949. / 16384, -185. / 16384, 35. / 32768.};
    }

    template <>
    inline std::array<double, 5> prediction_coeffs<5>()
    {
        return {-29011. / 131072., 569. / 8192., -4661. / 262144., 49. / 16384., -63. / 262144.};
    }

    template <class T>
    struct field_hack
    {
        field_hack(T&& t)
            : m_e{std::forward<T>(t)}
        {
        }

        template <std::size_t s, class interval_t, class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, s>, std::size_t level, const interval_t& i, const index_t... index)
        {
            return m_e(level, i, index...);
        }

        T m_e;
    };

    template <class T>
    inline auto make_field_hack(T&& t)
    {
        return field_hack<T>(std::forward<T>(t));
    }

    template <std::size_t S, class T, class C>
    struct Qs_i_impl
    {
        static constexpr std::size_t slim = S;

        inline Qs_i_impl(T&& e, C& c)
            : m_e{std::forward<T>(e)}
            , m_c{c}
        {
        }

        template <std::size_t s, class interval_t, class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, s>, std::size_t level, const interval_t& i, const index_t... index)
        {
            using coord_index_t = typename interval_t::coord_index_t;
            return xt::eval(m_c[s - 1]
                                * (m_e(std::integral_constant<std::size_t, s + 1>{}, level, i + static_cast<coord_index_t>(s), index...)
                                   - m_e(std::integral_constant<std::size_t, s + 1>{}, level, i - static_cast<coord_index_t>(s), index...))
                            + operator()(std::integral_constant<std::size_t, s + 1>{}, level, i, index...));
        }

        template <class interval_t, class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, slim>, std::size_t level, const interval_t& i, const index_t... index)
        {
            using coord_index_t = typename interval_t::coord_index_t;
            return xt::eval(m_c[slim - 1]
                            * (m_e(std::integral_constant<std::size_t, slim>{}, level, i + static_cast<coord_index_t>(slim), index...)
                               - m_e(std::integral_constant<std::size_t, slim>{}, level, i - static_cast<coord_index_t>(slim), index...)));
        }

        T m_e;
        C m_c;
    };

    template <std::size_t S, class T, class C>
    inline auto make_Qs_i(T&& t, C&& c)
    {
        return Qs_i_impl<S, T, C>(std::forward<T>(t), std::forward<C>(c));
    }

    template <std::size_t s, class Field, class interval_t, class... index_t>
    inline auto Qs_i(const Field& field, std::size_t level, const interval_t& i, const index_t... index)
    {
        auto c  = prediction_coeffs<s>();
        auto qs = make_Qs_i<s>(make_field_hack(field), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, index...);
    }

    template <std::size_t S, class T, class C>
    struct Qs_j_impl
    {
        static constexpr std::size_t slim = S;

        inline Qs_j_impl(T&& e, C& c)
            : m_e{std::forward<T>(e)}
            , m_c{c}
        {
        }

        template <std::size_t s, class interval_t, class coord_index_t = typename interval_t::coord_index_t, class... index_t>
        inline auto
        operator()(std::integral_constant<std::size_t, s>, std::size_t level, const interval_t& i, const coord_index_t j, const index_t... index)
        {
            return xt::eval(m_c[s - 1]
                                * (m_e(std::integral_constant<std::size_t, s + 1>{}, level, i, j + static_cast<coord_index_t>(s), index...)
                                   - m_e(std::integral_constant<std::size_t, s + 1>{}, level, i, j - static_cast<coord_index_t>(s), index...))
                            + operator()(std::integral_constant<std::size_t, s + 1>{}, level, i, j, index...));
        }

        template <class interval_t, class coord_index_t = typename interval_t::coord_index_t, class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, slim>,
                               std::size_t level,
                               const interval_t& i,
                               const coord_index_t j,
                               const index_t... index)
        {
            return xt::eval(m_c[slim - 1]
                            * (m_e(std::integral_constant<std::size_t, slim>{}, level, i, j + static_cast<coord_index_t>(slim), index...)
                               - m_e(std::integral_constant<std::size_t, slim>{}, level, i, j - static_cast<coord_index_t>(slim), index...)));
        }

        T m_e;
        C m_c;
    };

    template <std::size_t S, class T, class C>
    inline auto make_Qs_j(T&& t, C&& c)
    {
        return Qs_j_impl<S, T, C>(std::forward<T>(t), std::forward<C>(c));
    }

    template <std::size_t s, class Field, class interval_t, class coord_index_t = typename interval_t::coord_index_t, class... index_t>
    inline auto Qs_j(const Field& field, std::size_t level, const interval_t& i, const coord_index_t j, const index_t... index)
    {
        auto c  = prediction_coeffs<s>();
        auto qs = make_Qs_j<s>(make_field_hack(field), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, index...);
    }

    template <std::size_t S, class T, class C>
    struct Qs_k_impl
    {
        static constexpr std::size_t slim = S;

        inline Qs_k_impl(T&& e, C& c)
            : m_e{std::forward<T>(e)}
            , m_c{c}
        {
        }

        template <std::size_t s, class interval_t, class coord_index_t = typename interval_t::coord_index_t>
        inline auto
        operator()(std::integral_constant<std::size_t, s>, std::size_t level, const interval_t& i, const coord_index_t j, const coord_index_t k)
        {
            return xt::eval(m_c[s - 1]
                                * (m_e(std::integral_constant<std::size_t, s + 1>{}, level, i, j, k + static_cast<coord_index_t>(s))
                                   - m_e(std::integral_constant<std::size_t, s + 1>{}, level, i, j, k - static_cast<coord_index_t>(s)))
                            + operator()(std::integral_constant<std::size_t, s + 1>{}, level, i, j, k));
        }

        template <class interval_t, class coord_index_t = typename interval_t::coord_index_t>
        inline auto operator()(std::integral_constant<std::size_t, slim>,
                               std::size_t level,
                               const interval_t& i,
                               const coord_index_t j,
                               const coord_index_t k)
        {
            return xt::eval(m_c[slim - 1]
                            * (m_e(std::integral_constant<std::size_t, slim>{}, level, i, j, k + static_cast<coord_index_t>(slim))
                               - m_e(std::integral_constant<std::size_t, slim>{}, level, i, j, k - static_cast<coord_index_t>(slim))));
        }

        T m_e;
        C m_c;
    };

    template <std::size_t S, class T, class C>
    inline auto make_Qs_k(T&& t, C&& c)
    {
        return Qs_k_impl<S, T, C>(std::forward<T>(t), std::forward<C>(c));
    }

    template <std::size_t s, class Field, class interval_t, class coord_index_t = typename interval_t::coord_index_t>
    inline auto Qs_k(const Field& field, std::size_t level, const interval_t& i, const coord_index_t j, const coord_index_t k)
    {
        auto c  = prediction_coeffs<s>();
        auto qs = make_Qs_k<s>(make_field_hack(field), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, k);
    }

    template <std::size_t s, class Field, class interval_t, class coord_index_t = typename interval_t::coord_index_t, class... index_t>
    inline auto Qs_ij(const Field& field, std::size_t level, const interval_t& i, const coord_index_t j, const index_t... index)
    {
        auto c  = prediction_coeffs<s>();
        auto qs = make_Qs_i<s>(make_Qs_j<s>(make_field_hack(field), c), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, index...);
    }

    template <std::size_t s, class Field, class interval_t, class coord_index_t = typename interval_t::coord_index_t, class... index_t>
    inline auto Qs_ik(const Field& field, std::size_t level, const interval_t& i, const coord_index_t j, const index_t... index)
    {
        auto c  = prediction_coeffs<s>();
        auto qs = make_Qs_i<s>(make_Qs_k<s>(make_field_hack(field), c), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, index...);
    }

    template <std::size_t s, class Field, class interval_t, class coord_index_t = typename interval_t::coord_index_t, class... index_t>
    inline auto Qs_jk(const Field& field, std::size_t level, const interval_t& i, const coord_index_t j, const index_t... index)
    {
        auto c  = prediction_coeffs<s>();
        auto qs = make_Qs_j<s>(make_Qs_k<s>(make_field_hack(field), c), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, index...);
    }

    template <std::size_t s, class Field, class interval_t, class coord_index_t = typename interval_t::coord_index_t>
    inline auto Qs_ijk(const Field& field, std::size_t level, const interval_t& i, const coord_index_t j, const coord_index_t k)
    {
        auto c  = prediction_coeffs<s>();
        auto qs = make_Qs_i<s>(make_Qs_j<s>(make_Qs_k<s>(make_field_hack(field), c), c), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, k);
    }

    /////////////////////////
    // prediction operator //
    /////////////////////////

    template <class TInterval>
    class prediction_op : public field_operator_base<TInterval>
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

    template <class TInterval>
    template <class T1, class T2>
    inline void prediction_op<TInterval>::operator()(Dim<1>,
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

    template <class TInterval>
    template <class T1, class T2>
    inline void prediction_op<TInterval>::operator()(Dim<1>,
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

    template <class TInterval>
    template <class T1, class T2, std::size_t order>
    inline void prediction_op<TInterval>::operator()(Dim<1>,
                                                     T1& dest,
                                                     const T2& src,
                                                     std::integral_constant<std::size_t, order>,
                                                     std::integral_constant<bool, true>) const
    {
        auto ii = i << 1;
        ii.step = 2;

        auto qs_i = Qs_i<order>(src, level, i);

        dest(level + 1, ii)     = src(level, i) + qs_i;
        dest(level + 1, ii + 1) = src(level, i) - qs_i;
    }

    template <class TInterval>
    template <class T1, class T2, std::size_t order>
    inline void prediction_op<TInterval>::operator()(Dim<1>,
                                                     T1& dest,
                                                     const T2& src,
                                                     std::integral_constant<std::size_t, order>,
                                                     std::integral_constant<bool, false>) const
    {
        auto qs_i = Qs_i<order>(src, level - 1, i >> 1);

        auto even_i = i.even_elements();
        if (even_i.is_valid())
        {
            auto coarse_even_i  = even_i >> 1;
            auto dec_even       = (i.start & 1) ? 1 : 0;
            dest(level, even_i) = src(level - 1, coarse_even_i) + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]));
        }

        auto odd_i = i.odd_elements();
        if (odd_i.is_valid())
        {
            auto coarse_odd_i  = odd_i >> 1;
            auto dec_odd       = (i.end & 1) ? 1 : 0;
            dest(level, odd_i) = src(level - 1, coarse_odd_i) - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)));
        }
    }

    template <class TInterval>
    template <class T1, class T2>
    inline void prediction_op<TInterval>::operator()(Dim<2>,
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

    template <class TInterval>
    template <class T1, class T2>
    inline void prediction_op<TInterval>::operator()(Dim<2>,
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

    template <class TInterval>
    template <class T1, class T2, std::size_t order>
    inline void prediction_op<TInterval>::operator()(Dim<2>,
                                                     T1& dest,
                                                     const T2& src,
                                                     std::integral_constant<std::size_t, order>,
                                                     std::integral_constant<bool, true>) const
    {
        auto ii = i << 1;
        ii.step = 2;

        auto jj = j << 1;

        auto qs_i  = Qs_i<order>(src, level, i, j);
        auto qs_j  = Qs_j<order>(src, level, i, j);
        auto qs_ij = Qs_ij<order>(src, level, i, j);

        dest(level + 1, ii, jj)         = src(level, i, j) + qs_i + qs_j - qs_ij;
        dest(level + 1, ii + 1, jj)     = src(level, i, j) - qs_i + qs_j + qs_ij;
        dest(level + 1, ii, jj + 1)     = src(level, i, j) + qs_i - qs_j + qs_ij;
        dest(level + 1, ii + 1, jj + 1) = src(level, i, j) - qs_i - qs_j - qs_ij;
    }

    template <class TInterval>
    template <class T1, class T2, std::size_t order>
    inline void prediction_op<TInterval>::operator()(Dim<2>,
                                                     T1& dest,
                                                     const T2& src,
                                                     std::integral_constant<std::size_t, order>,
                                                     std::integral_constant<bool, false>) const
    {
        auto qs_i  = Qs_i<order>(src, level - 1, i >> 1, j >> 1);
        auto qs_j  = Qs_j<order>(src, level - 1, i >> 1, j >> 1);
        auto qs_ij = Qs_ij<order>(src, level - 1, i >> 1, j >> 1);

        if (j & 1)
        {
            auto even_i = i.even_elements();
            if (even_i.is_valid())
            {
                auto coarse_even_i = even_i >> 1;
                auto dec_even      = (i.start & 1) ? 1 : 0;
                if constexpr (T1::is_soa && T1::size > 1)
                {
                    dest(level, even_i, j) = src(level - 1, coarse_even_i, j >> 1)
                                           + xt::view(qs_i, xt::all(), xt::range(dec_even, qs_i.shape()[1]))
                                           - xt::view(qs_j, xt::all(), xt::range(dec_even, qs_j.shape()[1]))
                                           + xt::view(qs_ij, xt::all(), xt::range(dec_even, qs_ij.shape()[1]));
                }
                else
                {
                    dest(level, even_i, j) = src(level - 1, coarse_even_i, j >> 1) + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]))
                                           - xt::view(qs_j, xt::range(dec_even, qs_j.shape()[0]))
                                           + xt::view(qs_ij, xt::range(dec_even, qs_ij.shape()[0]));
                }
            }

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
            {
                auto coarse_odd_i = odd_i >> 1;
                auto dec_odd      = (i.end & 1) ? 1 : 0;
                if constexpr (T1::is_soa && T1::size > 1)
                {
                    dest(level, odd_i, j) = src(level - 1, coarse_odd_i, j >> 1)
                                          - xt::view(qs_i, xt::all(), xt::range(0, safe_subs<int>(qs_i.shape()[1], dec_odd)))
                                          - xt::view(qs_j, xt::all(), xt::range(0, safe_subs<int>(qs_j.shape()[1], dec_odd)))
                                          - xt::view(qs_ij, xt::all(), xt::range(0, safe_subs<int>(qs_ij.shape()[1], dec_odd)));
                }
                else
                {
                    dest(level, odd_i, j) = src(level - 1, coarse_odd_i, j >> 1)
                                          - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)))
                                          - xt::view(qs_j, xt::range(0, safe_subs<int>(qs_j.shape()[0], dec_odd)))
                                          - xt::view(qs_ij, xt::range(0, safe_subs<int>(qs_ij.shape()[0], dec_odd)));
                }
            }
        }
        else
        {
            auto even_i = i.even_elements();
            if (even_i.is_valid())
            {
                auto coarse_even_i = even_i >> 1;
                auto dec_even      = (i.start & 1) ? 1 : 0;
                if constexpr (T1::is_soa && T1::size > 1)
                {
                    dest(level, even_i, j) = src(level - 1, coarse_even_i, j >> 1)
                                           + xt::view(qs_i, xt::all(), xt::range(dec_even, qs_i.shape()[1]))
                                           + xt::view(qs_j, xt::all(), xt::range(dec_even, qs_j.shape()[1]))
                                           - xt::view(qs_ij, xt::all(), xt::range(dec_even, qs_ij.shape()[1]));
                }
                else
                {
                    dest(level, even_i, j) = src(level - 1, coarse_even_i, j >> 1) + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]))
                                           + xt::view(qs_j, xt::range(dec_even, qs_j.shape()[0]))
                                           - xt::view(qs_ij, xt::range(dec_even, qs_ij.shape()[0]));
                }
            }

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
            {
                auto coarse_odd_i = odd_i >> 1;
                auto dec_odd      = (i.end & 1) ? 1 : 0;
                if constexpr (T1::is_soa && T1::size > 1)
                {
                    dest(level, odd_i, j) = src(level - 1, coarse_odd_i, j >> 1)
                                          - xt::view(qs_i, xt::all(), xt::range(0, safe_subs<int>(qs_i.shape()[1], dec_odd)))
                                          + xt::view(qs_j, xt::all(), xt::range(0, safe_subs<int>(qs_j.shape()[1], dec_odd)))
                                          + xt::view(qs_ij, xt::all(), xt::range(0, safe_subs<int>(qs_ij.shape()[1], dec_odd)));
                }
                else
                {
                    dest(level, odd_i, j) = src(level - 1, coarse_odd_i, j >> 1)
                                          - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)))
                                          + xt::view(qs_j, xt::range(0, safe_subs<int>(qs_j.shape()[0], dec_odd)))
                                          + xt::view(qs_ij, xt::range(0, safe_subs<int>(qs_ij.shape()[0], dec_odd)));
                }
            }
        }
    }

    template <class TInterval>
    template <class T1, class T2>
    inline void prediction_op<TInterval>::operator()(Dim<3>,
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

    template <class TInterval>
    template <class T1, class T2>
    inline void prediction_op<TInterval>::operator()(Dim<3>,
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

    template <class TInterval>
    template <class T1, class T2, std::size_t order>
    inline void prediction_op<TInterval>::operator()(Dim<3>,
                                                     T1& dest,
                                                     const T2& src,
                                                     std::integral_constant<std::size_t, order>,
                                                     std::integral_constant<bool, true>) const
    {
        auto ii = i << 1;
        ii.step = 2;

        auto jj = j << 1;
        auto kk = k << 1;

        auto qs_i   = Qs_i<order>(src, level, i, j, k);
        auto qs_j   = Qs_j<order>(src, level, i, j, k);
        auto qs_k   = Qs_k<order>(src, level, i, j, k);
        auto qs_ij  = Qs_ij<order>(src, level, i, j, k);
        auto qs_ik  = Qs_ik<order>(src, level, i, j, k);
        auto qs_jk  = Qs_jk<order>(src, level, i, j, k);
        auto qs_ijk = Qs_ijk<order>(src, level, i, j, k);

        dest(level + 1, ii, jj, kk)             = src(level, i, j, k) + qs_i + qs_j + qs_k - qs_ij - qs_ik - qs_jk + qs_ijk;
        dest(level + 1, ii + 1, jj, kk)         = src(level, i, j, k) - qs_i + qs_j + qs_k + qs_ij + qs_ik - qs_jk - qs_ijk;
        dest(level + 1, ii, jj + 1, kk)         = src(level, i, j, k) + qs_i - qs_j + qs_k + qs_ij - qs_ik + qs_jk - qs_ijk;
        dest(level + 1, ii + 1, jj + 1, kk)     = src(level, i, j, k) - qs_i - qs_j + qs_k - qs_ij + qs_ik + qs_jk + qs_ijk;
        dest(level + 1, ii, jj, kk + 1)         = src(level, i, j, k) + qs_i + qs_j - qs_k - qs_ij + qs_ik + qs_jk - qs_ijk;
        dest(level + 1, ii + 1, jj, kk + 1)     = src(level, i, j, k) - qs_i + qs_j - qs_k + qs_ij - qs_ik + qs_jk + qs_ijk;
        dest(level + 1, ii, jj + 1, kk + 1)     = src(level, i, j, k) + qs_i - qs_j - qs_k + qs_ij + qs_ik - qs_jk + qs_ijk;
        dest(level + 1, ii + 1, jj + 1, kk + 1) = src(level, i, j, k) - qs_i - qs_j - qs_k - qs_ij - qs_ik - qs_jk - qs_ijk;
    }

    template <class TInterval>
    template <class T1, class T2, std::size_t order>
    inline void prediction_op<TInterval>::operator()(Dim<3>,
                                                     T1& dest,
                                                     const T2& src,
                                                     std::integral_constant<std::size_t, order>,
                                                     std::integral_constant<bool, false>) const
    {
        auto qs_i   = Qs_i<order>(src, level - 1, i >> 1, j >> 1, k >> 1);
        auto qs_j   = Qs_j<order>(src, level - 1, i >> 1, j >> 1, k >> 1);
        auto qs_k   = Qs_k<order>(src, level - 1, i >> 1, j >> 1, k >> 1);
        auto qs_ij  = Qs_ij<order>(src, level - 1, i >> 1, j >> 1, k >> 1);
        auto qs_ik  = Qs_ik<order>(src, level - 1, i >> 1, j >> 1, k >> 1);
        auto qs_jk  = Qs_jk<order>(src, level - 1, i >> 1, j >> 1, k >> 1);
        auto qs_ijk = Qs_ijk<order>(src, level - 1, i >> 1, j >> 1, k >> 1);

        if (k & 1)
        {
            if (j & 1)
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    auto coarse_even_i        = even_i >> 1;
                    auto dec_even             = (i.start & 1) ? 1 : 0;
                    dest(level, even_i, j, k) = src(level - 1, coarse_even_i, j >> 1, k >> 1)
                                              + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]))
                                              - xt::view(qs_j, xt::range(dec_even, qs_j.shape()[0]))
                                              - xt::view(qs_k, xt::range(dec_even, qs_k.shape()[0]))
                                              + xt::view(qs_ij, xt::range(dec_even, qs_ij.shape()[0]))
                                              + xt::view(qs_ik, xt::range(dec_even, qs_ik.shape()[0]))
                                              - xt::view(qs_jk, xt::range(dec_even, qs_jk.shape()[0]))
                                              + xt::view(qs_ijk, xt::range(dec_even, qs_ijk.shape()[0]));
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    auto coarse_odd_i        = odd_i >> 1;
                    auto dec_odd             = (i.end & 1) ? 1 : 0;
                    dest(level, odd_i, j, k) = src(level - 1, coarse_odd_i, j >> 1, k >> 1)
                                             - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)))
                                             - xt::view(qs_j, xt::range(0, safe_subs<int>(qs_j.shape()[0], dec_odd)))
                                             - xt::view(qs_k, xt::range(0, safe_subs<int>(qs_k.shape()[0], dec_odd)))
                                             - xt::view(qs_ij, xt::range(0, safe_subs<int>(qs_ij.shape()[0], dec_odd)));
                    -xt::view(qs_ik, xt::range(0, safe_subs<int>(qs_ik.shape()[0], dec_odd)));
                    -xt::view(qs_jk, xt::range(0, safe_subs<int>(qs_jk.shape()[0], dec_odd)));
                    -xt::view(qs_ijk, xt::range(0, safe_subs<int>(qs_ijk.shape()[0], dec_odd)));
                }
            }
            else
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    auto coarse_even_i        = even_i >> 1;
                    auto dec_even             = (i.start & 1) ? 1 : 0;
                    dest(level, even_i, j, k) = src(level - 1, coarse_even_i, j >> 1, k >> 1)
                                              + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]))
                                              + xt::view(qs_j, xt::range(dec_even, qs_j.shape()[0]))
                                              - xt::view(qs_k, xt::range(dec_even, qs_k.shape()[0]))
                                              - xt::view(qs_ij, xt::range(dec_even, qs_ij.shape()[0]))
                                              + xt::view(qs_ik, xt::range(dec_even, qs_ik.shape()[0]))
                                              + xt::view(qs_jk, xt::range(dec_even, qs_jk.shape()[0]))
                                              - xt::view(qs_ijk, xt::range(dec_even, qs_ijk.shape()[0]));
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    auto coarse_odd_i        = odd_i >> 1;
                    auto dec_odd             = (i.end & 1) ? 1 : 0;
                    dest(level, odd_i, j, k) = src(level - 1, coarse_odd_i, j >> 1, k >> 1)
                                             - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)))
                                             + xt::view(qs_j, xt::range(0, safe_subs<int>(qs_j.shape()[0], dec_odd)))
                                             - xt::view(qs_k, xt::range(0, safe_subs<int>(qs_k.shape()[0], dec_odd)))
                                             + xt::view(qs_ij, xt::range(0, safe_subs<int>(qs_ij.shape()[0], dec_odd)));
                    -xt::view(qs_ik, xt::range(0, safe_subs<int>(qs_ik.shape()[0], dec_odd)));
                    +xt::view(qs_jk, xt::range(0, safe_subs<int>(qs_jk.shape()[0], dec_odd)));
                    +xt::view(qs_ijk, xt::range(0, safe_subs<int>(qs_ijk.shape()[0], dec_odd)));
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
                    auto coarse_even_i        = even_i >> 1;
                    auto dec_even             = (i.start & 1) ? 1 : 0;
                    dest(level, even_i, j, k) = src(level - 1, coarse_even_i, j >> 1, k >> 1)
                                              + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]))
                                              - xt::view(qs_j, xt::range(dec_even, qs_j.shape()[0]))
                                              + xt::view(qs_k, xt::range(dec_even, qs_k.shape()[0]))
                                              + xt::view(qs_ij, xt::range(dec_even, qs_ij.shape()[0]))
                                              - xt::view(qs_ik, xt::range(dec_even, qs_ik.shape()[0]))
                                              + xt::view(qs_jk, xt::range(dec_even, qs_jk.shape()[0]))
                                              - xt::view(qs_ijk, xt::range(dec_even, qs_ijk.shape()[0]));
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    auto coarse_odd_i        = odd_i >> 1;
                    auto dec_odd             = (i.end & 1) ? 1 : 0;
                    dest(level, odd_i, j, k) = src(level - 1, coarse_odd_i, j >> 1, k >> 1)
                                             - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)))
                                             - xt::view(qs_j, xt::range(0, safe_subs<int>(qs_j.shape()[0], dec_odd)))
                                             + xt::view(qs_k, xt::range(0, safe_subs<int>(qs_k.shape()[0], dec_odd)))
                                             - xt::view(qs_ij, xt::range(0, safe_subs<int>(qs_ij.shape()[0], dec_odd)));
                    +xt::view(qs_ik, xt::range(0, safe_subs<int>(qs_ik.shape()[0], dec_odd)));
                    +xt::view(qs_jk, xt::range(0, safe_subs<int>(qs_jk.shape()[0], dec_odd)));
                    +xt::view(qs_ijk, xt::range(0, safe_subs<int>(qs_ijk.shape()[0], dec_odd)));
                }
            }
            else
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    auto coarse_even_i        = even_i >> 1;
                    auto dec_even             = (i.start & 1) ? 1 : 0;
                    dest(level, even_i, j, k) = src(level - 1, coarse_even_i, j >> 1, k >> 1)
                                              + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]))
                                              + xt::view(qs_j, xt::range(dec_even, qs_j.shape()[0]))
                                              + xt::view(qs_k, xt::range(dec_even, qs_k.shape()[0]))
                                              - xt::view(qs_ij, xt::range(dec_even, qs_ij.shape()[0]))
                                              - xt::view(qs_ik, xt::range(dec_even, qs_ik.shape()[0]))
                                              - xt::view(qs_jk, xt::range(dec_even, qs_jk.shape()[0]))
                                              + xt::view(qs_ijk, xt::range(dec_even, qs_ijk.shape()[0]));
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    auto coarse_odd_i        = odd_i >> 1;
                    auto dec_odd             = (i.end & 1) ? 1 : 0;
                    dest(level, odd_i, j, k) = src(level - 1, coarse_odd_i, j >> 1, k >> 1)
                                             - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)))
                                             + xt::view(qs_j, xt::range(0, safe_subs<int>(qs_j.shape()[0], dec_odd)))
                                             + xt::view(qs_k, xt::range(0, safe_subs<int>(qs_k.shape()[0], dec_odd)))
                                             + xt::view(qs_ij, xt::range(0, safe_subs<int>(qs_ij.shape()[0], dec_odd)));
                    +xt::view(qs_ik, xt::range(0, safe_subs<int>(qs_ik.shape()[0], dec_odd)));
                    -xt::view(qs_jk, xt::range(0, safe_subs<int>(qs_jk.shape()[0], dec_odd)));
                    -xt::view(qs_ijk, xt::range(0, safe_subs<int>(qs_ijk.shape()[0], dec_odd)));
                }
            }
        }
    }

    template <class TInterval>
    class variadic_prediction_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(variadic_prediction_op)

        template <std::size_t d, std::size_t order, bool dest_on_level>
        inline void operator()(Dim<d>, std::integral_constant<std::size_t, order>, std::integral_constant<bool, dest_on_level>) const
        {
        }

        template <std::size_t order, bool dest_on_level, class Head, class... Tail>
        inline void operator()(Dim<1>,
                               std::integral_constant<std::size_t, order> o,
                               std::integral_constant<bool, dest_on_level> dest,
                               Head& source,
                               Tail&... sources) const
        {
            prediction_op<interval_t>(level, i)(Dim<1>{}, source, source, o, dest);
            this->operator()(Dim<1>{}, o, dest, sources...);
        }

        template <std::size_t order, bool dest_on_level, class Head, class... Tail>
        inline void operator()(Dim<2>,
                               std::integral_constant<std::size_t, order> o,
                               std::integral_constant<bool, dest_on_level> dest,
                               Head& source,
                               Tail&... sources) const
        {
            prediction_op<interval_t>(level, i, j)(Dim<2>{}, source, source, o, dest);
            this->operator()(Dim<2>{}, o, dest, sources...);
        }

        template <std::size_t order, bool dest_on_level, class Head, class... Tail>
        inline void operator()(Dim<3>,
                               std::integral_constant<std::size_t, order> o,
                               std::integral_constant<bool, dest_on_level> dest,
                               Head& source,
                               Tail&... sources) const
        {
            prediction_op<interval_t>(level, i, j, k)(Dim<3>{}, source, source, o, dest);
            this->operator()(Dim<3>{}, o, dest, sources...);
        }
    };

    template <std::size_t order, bool dest_on_level, class... T>
    inline auto variadic_prediction(T&&... fields)
    {
        return make_field_operator_function<variadic_prediction_op>(std::integral_constant<std::size_t, order>{},
                                                                    std::integral_constant<bool, dest_on_level>{},
                                                                    std::forward<T>(fields)...);
    }

    template <std::size_t order, bool dest_on_level, class T>
    inline auto prediction(T& field)
    {
        return make_field_operator_function<prediction_op>(field,
                                                           field,
                                                           std::integral_constant<std::size_t, order>{},
                                                           std::integral_constant<bool, dest_on_level>{});
    }

    template <std::size_t order, bool dest_on_level, class T1, class T2>
    inline auto prediction(T1& field_dest, const T2& field_src)
    {
        return make_field_operator_function<prediction_op>(field_dest,
                                                           field_src,
                                                           std::integral_constant<std::size_t, order>{},
                                                           std::integral_constant<bool, dest_on_level>{});
    }
}
