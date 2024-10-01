#pragma once
#include <math.h>
#include <type_traits>

namespace samurai
{
    namespace detail
    {
        template <class Container, class Enable = void>
        struct ScalarType
        {
            using Type = typename Container::value_type;
        };

        template <class Container>
        struct ScalarType<Container, typename std::enable_if_t<std::is_floating_point_v<Container>>>
        {
            using Type = Container;
        };
    }

    /**
     * WENO5 implementation.
     * Based on 'Efficent implementation of Weighted ENO schemes', Jiang and Shu, 1996.
     */
    template <class stencil_values>
    auto compute_weno5_flux(stencil_values& f)
    {
        using value_type = typename stencil_values::value_type;
        // using scalar_type = std::conditional_t<std::is_floating_point_v<value_type>, value_type, typename value_type::value_type>;
        using scalar_type = typename detail::ScalarType<value_type>::Type;

        static constexpr std::size_t j = 2; // stencil center in f
        const scalar_type eps          = 1e-6;

        // clang-format off

        // (2.8) and Table I (r=3)
        auto q0 =  1./3 * f[j-2] - 7./6 * f[j-1] + 11./6 * f[j  ];
        auto q1 = -1./6 * f[j-1] + 5./6 * f[j  ] +  1./3 * f[j+1];
        auto q2 =  1./3 * f[j  ] + 5./6 * f[j+1] -  1./6 * f[j+2];

        // (3.2)-(3.4)
        auto IS0 = 13./12 * pow(f[j-2] - 2*f[j-1] + f[j  ], 2) + 1./4 * pow(  f[j-2] -4*f[j-1] + 3*f[j  ], 2);
        auto IS1 = 13./12 * pow(f[j-1] - 2*f[j  ] + f[j+1], 2) + 1./4 * pow(  f[j-1]           -   f[j+1], 2);
        auto IS2 = 13./12 * pow(f[j  ] - 2*f[j+1] + f[j+2], 2) + 1./4 * pow(3*f[j  ] -4*f[j+1] +   f[j+2], 2);

        // clang-format on

        // (2.16) and Table II (r=3)
        auto alpha0 = 0.1 / pow((eps + IS0), 2);
        auto alpha1 = 0.6 / pow((eps + IS1), 2);
        auto alpha2 = 0.3 / pow((eps + IS2), 2);

        // (2.15)
        auto sum_alphas = alpha0 + alpha1 + alpha2;
        auto omega0     = alpha0 / sum_alphas;
        auto omega1     = alpha1 / sum_alphas;
        auto omega2     = alpha2 / sum_alphas;

        // (2.10)
        value_type flux = omega0 * q0 + omega1 * q1 + omega2 * q2;
        return flux;
    }

    // template <class ScalarType, class Field, class Func>
    // auto compute_weno5_flux(ScalarType velocity, const Field& u, Func&& continuous_flux)
    // {
    //     xt::xtensor_fixed<field_value_t, xt::xshape<5>> f;
    //     if (v >= 0)
    //     {
    //         f = {continuous_flux(velocity, u[cells[0]]), u[cells[1]], u[cells[2]], u[cells[3]], u[cells[4]]};
    //     }
    //     else
    //     {
    //         f = {u[cells[5]], u[cells[4]], u[cells[3]], u[cells[2]], u[cells[1]]};
    //     }
    //     f *= v;

    //     return compute_weno5_flux(f);
    // }

} // end namespace samurai
