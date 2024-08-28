// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <xtensor/xadapt.hpp>
#include <xtensor/xfunction.hpp>
#include <xtensor/xmasked_view.hpp>

#include "cell.hpp"
#include "field_expression.hpp"
#include "operators_base.hpp"

namespace samurai
{
    template <class D>
    class finite_volume : public field_expression<D>
    {
      public:

        using derived_type = D;

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

        // TODO:
        // - remove the eval calls. They are added to fix a bug with eigen

        template <class... CT>
        inline auto operator()(Dim<1>, CT&&... e) const
        {
            auto dx = detail::extract_mesh(std::forward<CT>(e)...).cell_length(derived_cast().level);
            return eval((derived_cast().right_flux(std::forward<CT>(e)...) - derived_cast().left_flux(std::forward<CT>(e)...)) / dx);
        }

        template <class... CT>
        inline auto operator()(Dim<2>, CT&&... e) const
        {
            auto dx = detail::extract_mesh(std::forward<CT>(e)...).cell_length(derived_cast().level);
            return eval((-derived_cast().left_flux(std::forward<CT>(e)...) + derived_cast().right_flux(std::forward<CT>(e)...)
                         + -derived_cast().down_flux(std::forward<CT>(e)...) + derived_cast().up_flux(std::forward<CT>(e)...))
                        / dx);
        }

        template <class... CT>
        inline auto operator()(Dim<3>, CT&&... e) const
        {
            auto dx = detail::extract_mesh(std::forward<CT>(e)...).cell_length(derived_cast().level);
            return eval((-derived_cast().left_flux(std::forward<CT>(e)...) + derived_cast().right_flux(std::forward<CT>(e)...)
                         + -derived_cast().down_flux(std::forward<CT>(e)...) + derived_cast().up_flux(std::forward<CT>(e)...)
                         + -derived_cast().front_flux(std::forward<CT>(e)...) + derived_cast().back_flux(std::forward<CT>(e)...))
                        / dx);
        }

      protected:

        finite_volume() = default;
    };

    template <class D>
    inline auto finite_volume<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto finite_volume<D>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D>
    inline auto finite_volume<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    /*******************
     * upwind operator *
     *******************/

    template <std::size_t dim, class TInterval>
    class upwind_op : public field_operator_base<dim, TInterval>,
                      public finite_volume<upwind_op<dim, TInterval>>
    {
      public:

        INIT_OPERATOR(upwind_op)

        template <class T1, class T2>
        inline auto flux(double a, T1&& ul, T2&& ur) const
        {
            // TODO(loic): remove the xt::eval (bug without, see
            // VF_advection_1d)
            return (.5 * a * (std::forward<T1>(ul) + std::forward<T2>(ur)) + .5 * std::abs(a) * (std::forward<T1>(ul) - std::forward<T2>(ur)));
        }

        // 1D
        template <class T>
        inline auto left_flux(double a, const T& u) const
        {
            return flux(a, u(level, i - 1), u(level, i));
        }

        template <class T>
        inline auto right_flux(double a, const T& u) const
        {
            return flux(a, u(level, i), u(level, i + 1));
        }

        // 2D
        template <class T>
        inline auto left_flux(std::array<double, 2> a, const T& u) const
        {
            return flux(a[0], u(level, i - 1, j), u(level, i, j));
        }

        template <class T>
        inline auto right_flux(std::array<double, 2> a, const T& u) const
        {
            return flux(a[0], u(level, i, j), u(level, i + 1, j));
        }

        template <class T>
        inline auto down_flux(std::array<double, 2> a, const T& u) const
        {
            return flux(a[1], u(level, i, j - 1), u(level, i, j));
        }

        template <class T>
        inline auto up_flux(std::array<double, 2> a, const T& u) const
        {
            return flux(a[1], u(level, i, j), u(level, i, j + 1));
        }

        // 3D
        template <class T>
        inline auto left_flux(std::array<double, 3> a, const T& u) const
        {
            return flux(a[0], u(level, i - 1, j, k), u(level, i, j, k));
        }

        template <class T>
        inline auto right_flux(std::array<double, 3> a, const T& u) const
        {
            return flux(a[0], u(level, i, j, k), u(level, i + 1, j, k));
        }

        template <class T>
        inline auto down_flux(std::array<double, 3> a, const T& u) const
        {
            return flux(a[1], u(level, i, j - 1, k), u(level, i, j, k));
        }

        template <class T>
        inline auto up_flux(std::array<double, 3> a, const T& u) const
        {
            return flux(a[1], u(level, i, j, k), u(level, i, j + 1, k));
        }

        template <class T>
        inline auto front_flux(std::array<double, 3> a, const T& u) const
        {
            return flux(a[2], u(level, i, j, k - 1), u(level, i, j, k));
        }

        template <class T>
        inline auto back_flux(std::array<double, 3> a, const T& u) const
        {
            return flux(a[2], u(level, i, j, k), u(level, i, j, k + 1));
        }
    };

    template <class... CT>
    inline auto upwind(CT&&... e)
    {
        return make_field_operator_function<upwind_op>(std::forward<CT>(e)...);
    }

    /*******************
     * upwind operator for the scalar Burgers equation *
     *******************/
    template <std::size_t dim, class TInterval>
    class upwind_scalar_burgers_op : public field_operator_base<dim, TInterval>,
                                     public finite_volume<upwind_scalar_burgers_op<dim, TInterval>>
    {
      public:

        INIT_OPERATOR(upwind_scalar_burgers_op)

        template <class T1, class T2>
        inline auto flux(double a, const T1& ul, const T2& ur) const
        {
            using namespace math;
            auto out = zeros_like(ul);

            auto mask1 = (a * ul) < (a * ur);
            auto mask2 = (ul * ur) > 0.0;

            auto min = eval(minimum(abs(ul), abs(ur)));
            auto max = eval(maximum(abs(ul), abs(ur)));

            apply_on_masked(mask1 && mask2,
                            [&](auto imask)
                            {
                                out(imask) = .5 * min(imask) * min(imask);
                            });

            apply_on_masked(!mask1,
                            [&](auto imask)
                            {
                                out(imask) = .5 * max(imask) * max(imask);
                            });

            return out;
        }

        // 2D
        template <class T>
        inline auto left_flux(std::array<double, 2> a, const T& u) const
        {
            return flux(a[0], u(level, i - 1, j), u(level, i, j));
        }

        template <class T>
        inline auto right_flux(std::array<double, 2> a, const T& u) const
        {
            return flux(a[0], u(level, i, j), u(level, i + 1, j));
        }

        template <class T>
        inline auto down_flux(std::array<double, 2> a, const T& u) const
        {
            return flux(a[1], u(level, i, j - 1), u(level, i, j));
        }

        template <class T>
        inline auto up_flux(std::array<double, 2> a, const T& u) const
        {
            return flux(a[1], u(level, i, j), u(level, i, j + 1));
        }
    };

    template <class... CT>
    inline auto upwind_scalar_burgers(CT&&... e)
    {
        return make_field_operator_function<upwind_scalar_burgers_op>(std::forward<CT>(e)...);
    }
} // namespace samurai
