// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include "xtl/xtype_traits.hpp"

#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"

#include "cell.hpp"
#include "interval.hpp"
#include "samurai_config.hpp"
#include "utils.hpp"

namespace samurai
{
    struct field_expression_tag
    {
    };

    template <class E>
    struct is_field_expression : std::is_same<xt::xexpression_tag_t<E>, field_expression_tag>
    {
    };

    template <class... E>
    struct field_comparable : xtl::conjunction<is_field_expression<E>...>
    {
    };

    template <class D>
    class field_expression : public xt::xexpression<D>
    {
      public:

        using expression_tag = field_expression_tag;
    };

    namespace detail
    {
        template <class E, class enable = void>
        struct xview_type_impl
        {
            using type = E;
        };

        template <class E>
        struct xview_type_impl<E, std::enable_if_t<is_field_expression<E>::value>>
        {
            using type = typename E::view_type;
        };
    } // namespace detail

    template <class E>
    using xview_type = detail::xview_type_impl<E>;

    template <class E>
    using xview_type_t = typename xview_type<E>::type;

    template <class F, class... CT>
    class field_function : public field_expression<field_function<F, CT...>>
    {
      public:

        using self_type    = field_function<F, CT...>;
        using functor_type = std::remove_reference_t<F>;

        using interval_t = default_config::interval_t; // TO BE FIX: Check the
                                                       // interval_t of each CT

        using expression_tag = field_expression_tag;

        template <class Func, class... CTA, class U = std::enable_if<!std::is_base_of<Func, self_type>::value>>
        field_function(Func&& f, CTA&&... e) noexcept;

        template <class... T>
        inline auto operator()(const std::size_t& level, const interval_t& interval, const T&... index) const
        {
            auto expr = evaluate(std::make_index_sequence<sizeof...(CT)>(), level, interval, index...);
            return expr;
        }

        template <std::size_t dim>
        inline auto operator()(const std::size_t& level,
                               const interval_t& interval,
                               const xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim - 1>>& index) const
        {
            auto expr = evaluate(std::make_index_sequence<sizeof...(CT)>(), level, interval, index);
            return expr;
        }

        template <std::size_t dim>
        inline auto operator()(const Cell<dim, interval_t>& cell) const
        {
            return evaluate(std::make_index_sequence<sizeof...(CT)>(), cell);
        }

        template <std::size_t... I, class... T>
        inline auto evaluate(std::index_sequence<I...>, T&&... t) const
        {
            return xt::eval(m_f(std::get<I>(m_e).operator()(std::forward<T>(t)...)...));
        }

        const auto& arguments() const
        {
            return m_e;
        }

      private:

        std::tuple<CT...> m_e;
        functor_type m_f;
    };

    template <class F, class... CT>
    template <class Func, class... CTA, class>
    inline field_function<F, CT...>::field_function(Func&& f, CTA&&... e) noexcept
        : m_e(std::forward<CTA>(e)...)
        , m_f(std::forward<Func>(f))
    {
    }

    template <class F, class... E>
    inline auto make_field_function(E&&... e) noexcept
    {
        using type = field_function<F, E...>;
        return type(F(), std::forward<E>(e)...);
    }
} // namespace samurai

namespace xt::detail
{
    template <class F, class... E>
    struct select_xfunction_expression<samurai::field_expression_tag, F, E...>
    {
        using type = samurai::field_function<F, E...>;
    };
} // namespace xt

namespace samurai
{
    // NOLINTBEGIN(misc-unused-using-decls)
    using xt::operator+;
    using xt::operator-;
    using xt::operator*;
    using xt::operator/;
    using xt::operator%;
    // NOLINTEND(misc-unused-using-decls)
} // namespace samurai
