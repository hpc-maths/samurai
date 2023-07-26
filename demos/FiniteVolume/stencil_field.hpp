// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/stencil_field.hpp>

namespace samurai
{
    template <class TInterval>
    class H_wrap_op : public field_operator_base<TInterval>,
                      public field_expression<H_wrap_op<TInterval>>
    {
      public:

        INIT_OPERATOR(H_wrap_op)

        template <class Field>
        inline auto operator()(Dim<2>, const Field& phi, const Field& phi_0, const std::size_t max_level) const
        {
            double dx = this->dx();

            auto out = xt::empty<double>({i.size()});

            out.fill(0.);

            if (level == max_level)
            {
                // // First order one sided
                // auto dxp = (phi(level, i + 1, j) - phi(level, i    , j))/dx;
                // auto dxm = (phi(level, i    , j) - phi(level, i - 1, j))/dx;

                // auto dyp = (phi(level, i, j + 1) - phi(level, i, j    ))/dx;
                // auto dym = (phi(level, i, j    ) - phi(level, i, j - 1))/dx;

                // // Second-order one sided
                auto dxp = 1. / dx * (.5 * phi(level, i - 2, j) - 2. * phi(level, i - 1, j) + 1.5 * phi(level, i, j));
                auto dxm = 1. / dx * (-.5 * phi(level, i + 2, j) + 2. * phi(level, i + 1, j) - 1.5 * phi(level, i, j));

                auto dyp = 1. / dx * (.5 * phi(level, i, j - 2) - 2. * phi(level, i, j - 1) + 1.5 * phi(level, i, j));
                auto dym = 1. / dx * (-.5 * phi(level, i, j + 2) + 2. * phi(level, i, j + 1) - 1.5 * phi(level, i, j));

                auto pos_part = [](auto& a)
                {
                    return xt::maximum(0., a);
                };

                auto neg_part = [](auto& a)
                {
                    return xt::minimum(0., a);
                };

                // auto out = xt::empty<double>({i.size()});

                xt::xtensor<bool, 1> mask = xt::sign(phi_0(level, i, j)) >= 0.;

                xt::masked_view(out, mask) = xt::sqrt(xt::maximum(xt::pow(pos_part(dxp), 2.), xt::pow(neg_part(dxm), 2.))
                                                      + xt::maximum(xt::pow(pos_part(dyp), 2.), xt::pow(neg_part(dym), 2.)))
                                           - 1.;

                xt::masked_view(out, !mask) = -(xt::sqrt(xt::maximum(xt::pow(neg_part(dxp), 2.), xt::pow(pos_part(dxm), 2.))
                                                         + xt::maximum(xt::pow(neg_part(dyp), 2.), xt::pow(pos_part(dym), 2.)))
                                                - 1.);

                // xt::xtensor<bool, 1> mask_close = xt::abs(phi_0(level, i, j))
                // > 10.*dx;
            }

            // xt::masked_view(out, mask_close) = 0.;

            return xt::eval(out);
        }
    };

    template <class... CT>
    inline auto H_wrap(CT&&... e)
    {
        return make_field_operator_function<H_wrap_op>(std::forward<CT>(e)...);
    }

    /*******************
     * upwind operator for the scalar advection equation with variable velocity
     *******************/

    template <class TInterval>
    class upwind_variable_op : public field_operator_base<TInterval>,
                               public finite_volume<upwind_variable_op<TInterval>>
    {
      public:

        INIT_OPERATOR(upwind_variable_op)

        template <class T0, class T1, class T2, class T3>
        inline auto flux(T0&& vel, T1&& ul, T2&& ur, double lb, T3&& r) const
        {
            // // Upwind
            // return xt::eval(.5*std::forward<T0>(vel)*(std::forward<T1>(ul) +
            // std::forward<T2>(ur)) +
            //         .5*xt::abs(std::forward<T0>(vel))*(std::forward<T1>(ul) -
            //         std::forward<T2>(ur)));

            // // Lax Wendroff

            //
            // return xt::eval(.5 * std::forward<T0>(vel) *
            // (std::forward<T1>(ul) + std::forward<T2>(ur))
            // -.5*lb*xt::pow(std::forward<T0>(vel), 2.)*(std::forward<T2>(ur)-std::forward<T1>(ul)));

            // // Lax Wendroff with minmod limiter

            // auto minmod = [](auto & y)
            // {
            //     return xt::maximum(0., xt::minimum(1., y));
            // };

            // auto superbee = [](auto & y)
            // {
            //     return xt::maximum(xt::maximum(xt::minimum(1., 2.*y),
            //     xt::minimum(2., y)), 0.);
            // };

            auto mc = [](auto& y)
            {
                return xt::maximum(0., xt::minimum(xt::minimum(2. * y, .5 * (1. + y)), 2.));
            };

            auto pos_part = [](auto& a)
            {
                return xt::maximum(0, a);
            };

            auto neg_part = [](auto& a)
            {
                return xt::minimum(0, a);
            };

            // return xt::eval((1. - minmod(std::forward<T3>(r))) *
            // (neg_part(std::forward<T0>(vel))*std::forward<T2>(ur)
            //                                    +pos_part(std::forward<T0>(vel))*std::forward<T1>(ul))
            //                                    // Upwind part of the flux
            //                     + minmod(std::forward<T3>(r)) * (.5 *
            //                     std::forward<T0>(vel) * (std::forward<T1>(ul)
            //                     + std::forward<T2>(ur))
            //                                 -.5*lb*xt::pow(std::forward<T0>(vel),
            //                                 2.)*(std::forward<T2>(ur)-std::forward<T1>(ul))));

            return xt::eval((neg_part(std::forward<T0>(vel)) * std::forward<T2>(ur)
                             + pos_part(std::forward<T0>(vel)) * std::forward<T1>(ul)) // Upwind part of the flux
                            + 0.5 * mc(std::forward<T3>(r)) * xt::abs(std::forward<T0>(vel)) * (1. - lb * xt::abs(std::forward<T0>(vel)))
                                  * (std::forward<T2>(ur) - std::forward<T1>(ul)));
        }

        // 2D
        template <class T0, class T1>
        inline auto left_flux(const T0& vel, const T1& u, double dt) const
        {
            // auto vel_at_interface = xt::eval(.5 * (vel(0, level, i-1, j) +
            // vel(0, level, i, j)));
            auto vel_at_interface = xt::eval(3. / 8 * vel(0, level, i - 1, j) + 3. / 4 * vel(0, level, i, j)
                                             - 1. / 8 * vel(0, level, i + 1, j));

            auto denom                   = xt::eval(u(level, i, j) - u(level, i - 1, j));
            auto mask                    = xt::abs(denom) < 1.e-8;
            xt::masked_view(denom, mask) = 1.e-8;

            xt::xtensor<bool, 1> mask_sign = vel_at_interface >= 0.;

            auto rm12                         = xt::eval(1. / denom);
            xt::masked_view(rm12, mask_sign)  = rm12 * (u(level, i - 1, j) - u(level, i - 2, j));
            xt::masked_view(rm12, !mask_sign) = rm12 * (u(level, i + 1, j) - u(level, i, j));

            return flux(vel_at_interface, u(level, i - 1, j), u(level, i, j), dt / dx(), rm12);
        }

        template <class T0, class T1>
        inline auto right_flux(const T0& vel, const T1& u, double dt) const
        {
            // auto vel_at_interface = xt::eval(.5 * (vel(0, level, i, j) +
            // vel(0, level, i+1, j)));
            auto vel_at_interface = xt::eval(3. / 8 * vel(0, level, i + 1, j) + 3. / 4 * vel(0, level, i, j)
                                             - 1. / 8 * vel(0, level, i - 1, j));

            auto denom                   = xt::eval(u(level, i + 1, j) - u(level, i, j));
            auto mask                    = xt::abs(denom) < 1.e-8;
            xt::masked_view(denom, mask) = 1.e-8;

            xt::xtensor<bool, 1> mask_sign = vel_at_interface >= 0.;

            auto rp12                         = xt::eval(1. / denom);
            xt::masked_view(rp12, mask_sign)  = rp12 * (u(level, i, j) - u(level, i - 1, j));
            xt::masked_view(rp12, !mask_sign) = rp12 * (u(level, i + 2, j) - u(level, i + 1, j));

            return flux(vel_at_interface, u(level, i, j), u(level, i + 1, j), dt / dx(), rp12);
        }

        template <class T0, class T1>
        inline auto down_flux(const T0& vel, const T1& u, double dt) const
        {
            // auto vel_at_interface = xt::eval(.5 * (vel(1, level, i, j-1) +
            // vel(1, level, i, j)));
            auto vel_at_interface = xt::eval(3. / 8 * vel(1, level, i, j - 1) + 3. / 4 * vel(1, level, i, j)
                                             - 1. / 8 * vel(1, level, i, j + 1));

            auto denom                   = xt::eval(u(level, i, j) - u(level, i, j - 1));
            auto mask                    = xt::abs(denom) < 1.e-8;
            xt::masked_view(denom, mask) = 1.e-8;

            xt::xtensor<bool, 1> mask_sign = vel_at_interface >= 0.;

            auto rm12                         = xt::eval(1. / denom);
            xt::masked_view(rm12, mask_sign)  = rm12 * (u(level, i, j - 1) - u(level, i, j - 2));
            xt::masked_view(rm12, !mask_sign) = rm12 * (u(level, i, j + 1) - u(level, i, j));

            return flux(vel_at_interface, u(level, i, j - 1), u(level, i, j), dt / dx(), rm12);
        }

        template <class T0, class T1>
        inline auto up_flux(const T0& vel, const T1& u, double dt) const
        {
            // auto vel_at_interface = xt::eval(.5 * (vel(1, level, i, j) +
            // vel(1, level, i, j+1)));
            auto vel_at_interface = xt::eval(3. / 8 * vel(1, level, i, j + 1) + 3. / 4 * vel(1, level, i, j)
                                             - 1. / 8 * vel(1, level, i, j - 1));

            auto denom                   = xt::eval(u(level, i, j + 1) - u(level, i, j));
            auto mask                    = xt::abs(denom) < 1.e-8;
            xt::masked_view(denom, mask) = 1.e-8;

            xt::xtensor<bool, 1> mask_sign = vel_at_interface >= 0.;

            auto rp12                         = xt::eval(1. / denom);
            xt::masked_view(rp12, mask_sign)  = rp12 * (u(level, i, j) - u(level, i, j - 1));
            xt::masked_view(rp12, !mask_sign) = rp12 * (u(level, i, j + 2) - u(level, i, j + 1));

            return flux(vel_at_interface, u(level, i, j), u(level, i, j + 1), dt / dx(), rp12);
        }
    };

    template <class... CT>
    inline auto upwind_variable(CT&&... e)
    {
        return make_field_operator_function<upwind_variable_op>(std::forward<CT>(e)...);
    }

    template <class TInterval>
    class upwind_Burgers_op : public field_operator_base<TInterval>,
                              public finite_volume<upwind_Burgers_op<TInterval>>
    {
      public:

        INIT_OPERATOR(upwind_Burgers_op)

        template <class T1, class T2>
        inline auto flux(T1&& ul, T2&& ur, double lb) const
        {
            return xt::eval(.5 * (.5 * xt::pow(std::forward<T1>(ul), 2.) + .5 * xt::pow(std::forward<T2>(ur), 2.))
                            - .5 * lb * (std::forward<T2>(ur) - std::forward<T1>(ul))); // Lax-Friedrichs
            // return xt::eval(0.5 * xt::pow(std::forward<T1>(ul), 2.)); //
            // Upwing - it works for positive solution
        }

        // 1D
        template <class T1>
        inline auto left_flux(const T1& u, double lb) const
        {
            return flux(u(level, i - 1), u(level, i), lb);
        }

        template <class T1>
        inline auto right_flux(const T1& u, double lb) const
        {
            return flux(u(level, i), u(level, i + 1), lb);
        }
    };

    template <class... CT>
    inline auto upwind_Burgers(CT&&... e)
    {
        return make_field_operator_function<upwind_Burgers_op>(std::forward<CT>(e)...);
    }

}
