// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/stencil_field.hpp>

namespace samurai
{
    template <std::size_t dim, class TInterval>
    class upwind_Burgers_op : public field_operator_base<dim, TInterval>,
                              public finite_volume<upwind_Burgers_op<dim, TInterval>>
    {
      public:

        INIT_OPERATOR(upwind_Burgers_op)

        template <class T1, class T2>
        SAMURAI_INLINE auto flux(T1&& ul, T2&& ur, double lb) const
        {
            using namespace math;
            return eval(.5 * (.5 * pow(std::forward<T1>(ul), 2.) + .5 * pow(std::forward<T2>(ur), 2.))
                        - .5 * lb * (std::forward<T2>(ur) - std::forward<T1>(ul))); // Lax-Friedrichs
            // return xt::eval(0.5 * xt::pow(std::forward<T1>(ul), 2.)); //
            // Upwing - it works for positive solution
        }

        // 1D
        template <class T1>
        SAMURAI_INLINE auto left_flux(const T1& u, double lb) const
        {
            // std::cout << "left flux " << level << " " << i << " " << lb << std::endl;
            // std::cout << flux(u(level, i - 1), u(level, i), lb) << std::endl;
            return flux(u(level, i - 1), u(level, i), lb);
        }

        template <class T1>
        SAMURAI_INLINE auto right_flux(const T1& u, double lb) const
        {
            // std::cout << flux(u(level, i), u(level, i + 1), lb) << std::endl;
            return flux(u(level, i), u(level, i + 1), lb);
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto upwind_Burgers(CT&&... e)
    {
        return make_field_operator_function<upwind_Burgers_op>(std::forward<CT>(e)...);
    }

}
