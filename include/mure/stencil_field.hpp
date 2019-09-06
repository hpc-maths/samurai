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

    /*******************
     * upwind operator *
     *******************/

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