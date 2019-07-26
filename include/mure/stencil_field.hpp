#pragma once

#include <xtensor/xfunction.hpp>

#include "cell.hpp"
#include "field_expression.hpp"
#include "operators.hpp"

namespace mure
{

    // struct stencil_field
    // {
    //     template<class coord_index_t, std::size_t dim>
    //     auto operator()(Cell<coord_index_t, dim> cell) const
    //     {
    //         std::cout << "coucou\n";
    //         return 3.;
    //     }

    //     auto operator()(double cell) const
    //     {
    //         std::cout << "coucou\n";
    //         return 1.;
    //     }
    // };

    // template<class E>
    // inline auto upwind(E &&e) noexcept
    // {
    //     return make_field_function<stencil_field>(std::forward<E>(e));
    // }

    // template<class E>
    // class stencil_field : public field_expression<stencil_field<E>> {
    //   public:
    //     using stencil = std::true_type;

    //     stencil_field(double a, field_expression<E> &e) : m_a{a}, m_e{e}
    //     {}

    //     template<class coord_index_t, std::size_t dim>
    //     auto operator()(const Cell<coord_index_t, dim> cell) const
    //     {
    //         return m_e.derived_cast()(cell);
    //     }

    //     template<class interval_t, class... Index>
    //     auto operator()(const std::size_t &level, const interval_t &interval,
    //                     const Index &... index) const
    //     {
    //         auto dx = 1. / (1 << level);
    //         return m_a / dx *
    //                (m_e.derived_cast()(level, interval + 1, index...) -
    //                 m_e.derived_cast()(level, interval, index...));
    //     }

    //   private:
    //     double m_a;
    //     field_expression<E> &m_e;
    // };

    // template<class E>
    // inline auto upwind(double a, E &&e)
    // {
    //     return stencil_field<std::decay_t<E>>(a, std::forward<E>(e));
    // }

    using left = std::integral_constant<std::size_t, 0>;
    using right = std::integral_constant<std::size_t, 1>;
    using down = std::integral_constant<std::size_t, 2>;
    using back = std::integral_constant<std::size_t, 3>;
    using front = std::integral_constant<std::size_t, 4>;

    // template<template<std::size_t Dim, class T> class OP, class... CT>
    // class finite_volume_scheme
    //     : public field_expression<finite_volume_scheme<OP, CT...>> {
    //   public:
    //     static constexpr std::size_t dim = detail::compute_dim<CT...>();

    //     finite_volume_scheme(CT &&... e) : m_e{std::forward<CT>(e)...}
    //     {}

    //     template<class interval_t, class... index_t>
    //     auto operator()(const std::size_t &level, const interval_t &i,
    //                     const index_t &... index) const
    //     {
    //         OP<dim, interval_t> op(level, i, index...);
    //         return apply(op);
    //     }

    //   private:
    //     template<class interval_t>
    //     auto apply(OP<dim, interval_t> &op) const
    //     {
    //         return apply_impl(std::make_index_sequence<sizeof...(CT)>(), op);
    //     }

    //     template<std::size_t... I, class interval_t>
    //     auto apply_impl(std::index_sequence<I...>,
    //                     OP<dim, interval_t> &op) const
    //     {
    //         return op(std::get<I>(m_e)...);
    //     }

    //     std::tuple<CT...> m_e;
    // };

    // template<template<std::size_t dim, class T> class OP, class... CT>
    // auto make_finite_volume_scheme(CT &&... e)
    // {
    //     return finite_volume_scheme<OP, CT...>(std::forward<CT>(e)...);
    // }

    // template<std::size_t dim, class TInterval>
    // class upwind_op {
    //     template<class T, std::size_t direction>
    //     auto flux(const T &u)
    //     {}
    // };

    // template<class TInterval>
    // class upwind_op<1> : public field_operator_base<TInterval> {
    //   public:
    //     using interval_t = TInterval;
    //     using coord_index_t = typename interval_t::coord_index_t;

    //     using base = field_operator_base<interval_t>;
    //     using base::i;
    //     using base::level;

    //     template<std::size_t dim>
    //     upwind_op(double a, std::size_t level,
    //               xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
    //               interval_t interval)
    //         : m_a{a}, base(level, index, interval)
    //     {}

    //     template<class T, std::size_t direction>
    //     auto flux(const T &u)
    //     {}

    //     template<class T>
    //     auto flux<left>(const T &u)
    //     {
    //         return a / dx * (u(level, i + 1) - u(level, i));
    //     }

    //     template<class T>
    //     auto flux<right>(const T &u)
    //     {
    //         return a / dx * (u(level, i) - u(level, i - 1));
    //     }

    //     template<class T>
    //     auto operator()(const T &u)
    //     {
    //         return flux<left>(u) + flux<right>(u);
    //     }
    // };

    template<class D>
    class finite_volume : public field_expression<D> {
      public:
        using derived_type = D;
        static constexpr std::size_t dim = D::dim;

        derived_type &derived_cast() & noexcept;
        const derived_type &derived_cast() const &noexcept;
        derived_type derived_cast() && noexcept;

        template<class... CT>
        auto operator()(Dim<1> d, CT &&... e) const
        {
            return derived_cast().left_flux(d, std::forward<CT>(e)...) +
                   derived_cast().right_flux(d, std::forward<CT>(e)...);
        }

        template<class... CT>
        auto operator()(Dim<2> d, CT &&... e) const
        {
            return derived_cast().left_flux(d, std::forward<CT>(e)...) +
                   derived_cast().right_flux(d, std::forward<CT>(e)...) +
                   derived_cast().down_flux(d, std::forward<CT>(e)...) +
                   derived_cast().up_flux(d, std::forward<CT>(e)...);
        }

      protected:
        finite_volume(){};
        ~finite_volume() = default;

        finite_volume(const finite_volume &) = default;
        finite_volume &operator=(const finite_volume &) = default;

        finite_volume(finite_volume &&) = default;
        finite_volume &operator=(finite_volume &&) = default;
    };

    template<class D>
        inline auto finite_volume<D>::derived_cast() &
        noexcept -> derived_type &
    {
        return *static_cast<derived_type *>(this);
    }

    template<class D>
        inline auto finite_volume<D>::derived_cast() const &
        noexcept -> const derived_type &
    {
        return *static_cast<const derived_type *>(this);
    }

    template<class D>
        inline auto finite_volume<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type *>(this);
    }

    template<class TInterval>
    class upwind_op : public field_operator_base<TInterval>,
                      public finite_volume<upwind_op<TInterval>> {
      public:
        INIT_OPERATOR(upwind_op)

        // 1D
        template<class T>
        auto left_flux(Dim<1>, double a, const T &u) const
        {
            return ((a < 0) ? a : 0) / dx * (u(level, i + 1) - u(level, i));
        }

        template<class T>
        auto right_flux(Dim<1>, double a, const T &u) const
        {
            return ((a > 0) ? a : 0) / dx * (u(level, i) - u(level, i - 1));
        }

        // 2D
        template<class T>
        auto left_flux(Dim<2>, std::array<double, 2> a, const T &u) const
        {
            return ((a[0] < 0) ? a[0] : 0) / dx *
                   (u(level, i + 1, j) - u(level, i, j));
        }

        template<class T>
        auto right_flux(Dim<2>, std::array<double, 2> a, const T &u) const
        {
            return ((a[0] > 0) ? a[0] : 0) / dx *
                   (u(level, i, j) - u(level, i - 1, j));
        }

        template<class T>
        auto down_flux(Dim<2>, std::array<double, 2> a, const T &u) const
        {
            return ((a[1] < 0) ? a[1] : 0) / dx *
                   (u(level, i, j + 1) - u(level, i, j));
        }

        template<class T>
        auto up_flux(Dim<2>, std::array<double, 2> a, const T &u) const
        {
            return ((a[1] > 0) ? a[1] : 0) / dx *
                   (u(level, i, j) - u(level, i, j - 1));
        }
    };

    template<class... CT>
    auto upwind(CT &&... e)
    {
        return make_field_operator_function<upwind_op>(std::forward<CT>(e)...);
    }
}