#pragma once

#include <xtensor/xfunction.hpp>

#include "cell.hpp"
#include "field_expression.hpp"
#include "operators_base.hpp"

namespace mure
{
    template<class D>
    class finite_volume : public field_expression<D> {
      public:
        using derived_type = D;

        derived_type &derived_cast() & noexcept;
        const derived_type &derived_cast() const &noexcept;
        derived_type derived_cast() && noexcept;

        template<class... CT>
        inline auto operator()(Dim<1> d, CT &&... e) const
        {
            return (derived_cast().right_flux(std::forward<CT>(e)...)
                   -derived_cast().left_flux(std::forward<CT>(e)...))/derived_cast().dx();
        }

        template<class... CT>
        inline auto operator()(Dim<2> d, CT &&... e) const
        {
            return (-derived_cast().left_flux(std::forward<CT>(e)...) +
                    derived_cast().right_flux(std::forward<CT>(e)...) +
                    -derived_cast().down_flux(std::forward<CT>(e)...) +
                    derived_cast().up_flux(std::forward<CT>(e)...))/derived_cast().dx();
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

    /*******************
     * upwind operator *
     *******************/

    template<class TInterval>
    class upwind_op : public field_operator_base<TInterval>,
                      public finite_volume<upwind_op<TInterval>> {
      public:
        INIT_OPERATOR(upwind_op)

        template<class T1, class T2>
        inline auto flux(double a, T1&& ul, T2&& ur) const
        {
            // TODO: rmeove the xt::eval (bug without, see VF_advection_1d)
            return (.5*a*(std::forward<T1>(ul) + std::forward<T2>(ur)) +
                    .5*std::abs(a)*(std::forward<T1>(ul) - std::forward<T2>(ur)));
        }

        // 1D
        template<class T>
        inline auto left_flux(double a, const T &u) const
        {
            return flux(a, u(level, i-1), u(level, i));

        }

        template<class T>
        inline auto right_flux(double a, const T &u) const
        {
            return flux(a, u(level, i), u(level, i+1));
        }

        // 2D
        template<class T>
        inline auto left_flux(std::array<double, 2> a, const T &u) const
        {
            return flux(a[0], u(level, i-1, j), u(level, i, j));
        }

        template<class T>
        inline auto right_flux(std::array<double, 2> a, const T &u) const
        {
            return flux(a[0], u(level, i, j), u(level, i+1, j));
        }

        template<class T>
        inline auto down_flux(std::array<double, 2> a, const T &u) const
        {
            return flux(a[1], u(level, i, j-1), u(level, i, j));
        }

        template<class T>
        inline auto up_flux(std::array<double, 2> a, const T &u) const
        {
            return flux(a[1], u(level, i, j), u(level, i, j+1));
        }
    };

    template<class... CT>
    inline auto upwind(CT &&... e)
    {
        return make_field_operator_function<upwind_op>(std::forward<CT>(e)...);
    }
}