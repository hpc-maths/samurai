// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>

namespace samurai
{
    template<std::size_t s>
    inline std::array<double, s> coeffs();

    template<>
    inline std::array<double, 1> coeffs<1>()
    {
        return {-1. / 8.};
    }

    template<>
    inline std::array<double, 2> coeffs<2>()
    {
        return {-22. / 128., 3. / 128.};
    }

    template<>
    inline std::array<double, 3> coeffs<3>()
    {
        return {-201. / 1024., 11. / 256., -5. / 1024.};
    }

    template<>
    inline std::array<double, 4> coeffs<4>()
    {
        return {-3461. / 16384., 949. / 16384, -185. / 16384, 35. / 32768.};
    }

    template<>
    inline std::array<double, 5> coeffs<5>()
    {
        return {-29011. / 131072., 569. / 8192., -4661. / 262144., 49. / 16384., -63. / 262144.};
    }

    template<class T>
    struct field_hack
    {
        field_hack(T &&t) : m_e{std::forward<T>(t)}
        {}

        template<std::size_t s, class interval_t, class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, s>,
                        std::size_t level,
                        const interval_t &i,
                        const index_t... index)
        {
            return m_e(level, i, index...);
        }

        T m_e;
    };

    template<class T>
    inline auto make_field_hack(T &&t)
    {
        return field_hack<T>(std::forward<T>(t));
    }

    template<std::size_t S, class T, class C>
    struct Qs_i_impl
    {
        static constexpr std::size_t slim = S;

        inline Qs_i_impl(T &&e, C &c) : m_e{std::forward<T>(e)}, m_c{c}
        {}

        template<std::size_t s, class interval_t, class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, s>,
                        std::size_t level,
                        const interval_t &i,
                        const index_t... index)
        {
            using coord_index_t = typename interval_t::coord_index_t;
            return xt::eval(m_c[s - 1] * (m_e(std::integral_constant<std::size_t, s + 1>{}, level,
                                              i + static_cast<coord_index_t>(s), index...) -
                                          m_e(std::integral_constant<std::size_t, s + 1>{}, level,
                                              i - static_cast<coord_index_t>(s), index...)) +
                            operator()(std::integral_constant<std::size_t, s + 1>{}, level, i, index...));
        }

        template<class interval_t, class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, slim>,
                        std::size_t level,
                        const interval_t &i,
                        const index_t... index)
        {
            using coord_index_t = typename interval_t::coord_index_t;
            return xt::eval(m_c[slim - 1] * (m_e(std::integral_constant<std::size_t, slim>{}, level,
                                                 i + static_cast<coord_index_t>(slim), index...) -
                                             m_e(std::integral_constant<std::size_t, slim>{}, level,
                                                 i - static_cast<coord_index_t>(slim), index...)));
        }

        T m_e;
        C m_c;
    };

    template<std::size_t S, class T, class C>
    inline auto make_Qs_i(T &&t, C &&c)
    {
        return Qs_i_impl<S, T, C>(std::forward<T>(t), std::forward<C>(c));
    }

    template<std::size_t s, class Field, class interval_t, class... index_t>
    inline auto Qs_i(const Field &field, std::size_t level, const interval_t &i, const index_t... index)
    {
        auto c = coeffs<s>();
        auto qs = make_Qs_i<s>(make_field_hack(field), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, index...);
    }

    template<std::size_t S, class T, class C>
    struct Qs_j_impl
    {
        static constexpr std::size_t slim = S;

        inline Qs_j_impl(T &&e, C &c) : m_e{std::forward<T>(e)}, m_c{c}
        {}

        template<std::size_t s,
                 class interval_t,
                 class coord_index_t = typename interval_t::coord_index_t,
                 class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, s>,
                        std::size_t level,
                        const interval_t &i,
                        const coord_index_t j,
                        const index_t... index)
        {
            return xt::eval(m_c[s - 1] * (m_e(std::integral_constant<std::size_t, s + 1>{}, level, i,
                                              j + static_cast<coord_index_t>(s), index...) -
                                          m_e(std::integral_constant<std::size_t, s + 1>{}, level, i,
                                              j - static_cast<coord_index_t>(s), index...)) +
                            operator()(std::integral_constant<std::size_t, s + 1>{}, level, i, j, index...));
        }

        template<class interval_t, class coord_index_t = typename interval_t::coord_index_t, class... index_t>
        inline auto operator()(std::integral_constant<std::size_t, slim>,
                        std::size_t level,
                        const interval_t &i,
                        const coord_index_t j,
                        const index_t... index)
        {
            return xt::eval(m_c[slim - 1] * (m_e(std::integral_constant<std::size_t, slim>{}, level, i,
                                                 j + static_cast<coord_index_t>(slim), index...) -
                                             m_e(std::integral_constant<std::size_t, slim>{}, level, i,
                                                 j - static_cast<coord_index_t>(slim), index...)));
        }

        T m_e;
        C m_c;
    };

    template<std::size_t S, class T, class C>
    inline auto make_Qs_j(T &&t, C &&c)
    {
        return Qs_j_impl<S, T, C>(std::forward<T>(t), std::forward<C>(c));
    }

    template<std::size_t s,
             class Field,
             class interval_t,
             class coord_index_t = typename interval_t::coord_index_t,
             class... index_t>
    inline auto Qs_j(const Field &field, std::size_t level, const interval_t &i, const coord_index_t j, const index_t... index)
    {
        auto c = coeffs<s>();
        auto qs = make_Qs_j<s>(make_field_hack(field), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, index...);
    }

    template<std::size_t S, class T, class C>
    struct Qs_k_impl
    {
        static constexpr std::size_t slim = S;

        inline Qs_k_impl(T &&e, C &c) : m_e{std::forward<T>(e)}, m_c{c}
        {}

        template<std::size_t s, class interval_t, class coord_index_t = typename interval_t::coord_index_t>
        inline auto operator()(std::integral_constant<std::size_t, s>,
                        std::size_t level,
                        const interval_t &i,
                        const coord_index_t j,
                        const coord_index_t k)
        {

            return xt::eval(m_c[s - 1] * (m_e(std::integral_constant<std::size_t, s + 1>{}, level, i, j,
                                              k + static_cast<coord_index_t>(s)) -
                                          m_e(std::integral_constant<std::size_t, s + 1>{}, level, i, j,
                                              k - static_cast<coord_index_t>(s))) +
                            operator()(std::integral_constant<std::size_t, s + 1>{}, level, i, j, k));
        }

        template<class interval_t, class coord_index_t = typename interval_t::coord_index_t>
        inline auto operator()(std::integral_constant<std::size_t, slim>,
                        std::size_t level,
                        const interval_t &i,
                        const coord_index_t j,
                        const coord_index_t k)
        {
            return xt::eval(
                m_c[slim - 1] *
                (m_e(std::integral_constant<std::size_t, slim>{}, level, i, j, k + static_cast<coord_index_t>(slim)) -
                 m_e(std::integral_constant<std::size_t, slim>{}, level, i, j, k - static_cast<coord_index_t>(slim))));
        }

        T m_e;
        C m_c;
    };

    template<std::size_t S, class T, class C>
    inline auto make_Qs_k(T &&t, C &&c)
    {
        return Qs_k_impl<S, T, C>(std::forward<T>(t), std::forward<C>(c));
    }

    template<std::size_t s, class Field, class interval_t, class coord_index_t = typename interval_t::coord_index_t>
    inline auto Qs_k(const Field &field, std::size_t level, const interval_t &i, const coord_index_t j, const coord_index_t k)
    {
        auto c = coeffs<s>();
        auto qs = make_Qs_k<s>(make_field_hack(field), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, k);
    }

    template<std::size_t s,
             class Field,
             class interval_t,
             class coord_index_t = typename interval_t::coord_index_t,
             class... index_t>
    inline auto
    Qs_ij(const Field &field, std::size_t level, const interval_t &i, const coord_index_t j, const index_t... index)
    {
        auto c = coeffs<s>();
        auto qs = make_Qs_i<s>(make_Qs_j<s>(make_field_hack(field), c), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, index...);
    }

    template<std::size_t s, class Field, class interval_t, class coord_index_t = typename interval_t::coord_index_t>
    inline auto
    Qs_ijk(const Field &field, std::size_t level, const interval_t &i, const coord_index_t j, const coord_index_t k)
    {
        auto c = coeffs<s>();
        auto qs = make_Qs_i<s>(make_Qs_j<s>(make_Qs_k<s>(make_field_hack(field), c), c), c);
        return qs(std::integral_constant<std::size_t, 1>{}, level, i, j, k);
    }
} // namespace samurai