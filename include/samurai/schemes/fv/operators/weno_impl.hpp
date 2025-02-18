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
    template <class flux_values_t, class stencil_values>
    inline auto compute_weno5_flux(flux_values_t& flux, stencil_values& f)
    {
        using value_type = typename stencil_values::value_type;
        // using scalar_type = std::conditional_t<std::is_floating_point_v<value_type>, value_type, typename value_type::value_type>;
        using scalar_type = typename detail::ScalarType<value_type>::Type;

        static constexpr std::size_t j = 2; // stencil center in f
        const scalar_type eps          = 1e-6;

        // clang-format off

        // (2.8) and Table I (r=3)
        auto q0 =  1./3. * f[j-2] - 7./6. * f[j-1] + 11./6. * f[j  ];
        auto q1 = -1./6. * f[j-1] + 5./6. * f[j  ] +  1./3. * f[j+1];
        auto q2 =  1./3. * f[j  ] + 5./6. * f[j+1] -  1./6. * f[j+2];

        // (3.2)-(3.4)
        auto IS0 = 13./12. * pow(f[j-2] - 2.*f[j-1] + f[j  ], 2) + 1./4. * pow(   f[j-2] -4.*f[j-1] + 3.*f[j  ], 2);
        auto IS1 = 13./12. * pow(f[j-1] - 2.*f[j  ] + f[j+1], 2) + 1./4. * pow(   f[j-1]            -    f[j+1], 2);
        auto IS2 = 13./12. * pow(f[j  ] - 2.*f[j+1] + f[j+2], 2) + 1./4. * pow(3.*f[j  ] -4.*f[j+1] +    f[j+2], 2);

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
        // value_type flux = omega0 * q0 + omega1 * q1 + omega2 * q2;
        // return flux;
        flux = omega0 * q0 + omega1 * q1 + omega2 * q2;
    }

    // template <class flux_values_t, class stencil_values>
    inline void
    compute_weno5_flux_vecto(double& flux, const double f_jm2, const double f_jm1, const double f_j, const double f_jp1, const double f_jp2)
    {
        // using value_type = typename stencil_values::value_type;
        // // using scalar_type = std::conditional_t<std::is_floating_point_v<value_type>, value_type, typename value_type::value_type>;
        // using scalar_type = typename detail::ScalarType<value_type>::Type;

        // static constexpr std::size_t j = 2; // stencil center in f
        // const scalar_type eps          = 1e-6;

        const double eps = 1e-6;

        // clang-format off

        // (2.8) and Table I (r=3)
        auto q0 =  1./3. * f_jm2 - 7./6. * f_jm1 + 11./6. * f_j;
        auto q1 = -1./6. * f_jm1 + 5./6. * f_j   +  1./3. * f_jp1;
        auto q2 =  1./3. * f_j   + 5./6. * f_jp1 -  1./6. * f_jp2;

        // (3.2)-(3.4)
        auto IS0 = 13./12. * pow(f_jm2 - 2.*f_jm1 + f_j  , 2) + 1./4. * pow(   f_jm2 -4.*f_jm1 + 3.*f_j  , 2);
        auto IS1 = 13./12. * pow(f_jm1 - 2.*f_j   + f_jp1, 2) + 1./4. * pow(   f_jm1           -    f_jp1, 2);
        auto IS2 = 13./12. * pow(f_j   - 2.*f_jp1 + f_jp2, 2) + 1./4. * pow(3.*f_j   -4.*f_jp1 +    f_jp2, 2);

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
        // value_type flux = omega0 * q0 + omega1 * q1 + omega2 * q2;
        // return flux;
        flux = omega0 * q0 + omega1 * q1 + omega2 * q2;
    }

    // template <class flux_values_t, class stencil_values, class tmp_variables_t>
    // inline void compute_weno5_flux__batch(flux_values_t& flux_values, const stencil_values& f, tmp_variables_t& tmp)
    // {
    //     using value_type = typename stencil_values::value_type;
    //     // using scalar_type = std::conditional_t<std::is_floating_point_v<value_type>, value_type, typename value_type::value_type>;
    //     using scalar_type = typename detail::ScalarType<value_type>::Type;

    //     static constexpr std::size_t j = 2; // stencil center in f
    //     const scalar_type eps          = 1e-6;

    //     auto batch_size = flux_values.size();
    //     assert(batch_size != 0);

    //     // flux_values_t q0(batch_size);
    //     // flux_values_t q1(batch_size);
    //     // flux_values_t q2(batch_size);

    //     // flux_values_t IS0(batch_size);
    //     // flux_values_t IS1(batch_size);
    //     // flux_values_t IS2(batch_size);
    //     auto& q0  = tmp.q0;
    //     auto& q1  = tmp.q1;
    //     auto& q2  = tmp.q2;
    //     auto& IS0 = tmp.IS0;
    //     auto& IS1 = tmp.IS1;
    //     auto& IS2 = tmp.IS2;

    //     for (std::size_t i = 0; i < batch_size; ++i)
    //     {
    //         // clang-format off

    //         // (2.8) and Table I (r=3)
    //         q0[i] =  1./3. * f[j-2][i] - 7./6. * f[j-1][i] + 11./6. * f[j  ][i];
    //         q1[i] = -1./6. * f[j-1][i] + 5./6. * f[j  ][i] +  1./3. * f[j+1][i];
    //         q2[i] =  1./3. * f[j  ][i] + 5./6. * f[j+1][i] -  1./6. * f[j+2][i];

    //         // (3.2)-(3.4)
    //         IS0[i] = 13./12. * pow(f[j-2][i] - 2.*f[j-1][i] + f[j  ][i], 2) + 1./4. * pow(   f[j-2][i] -4.*f[j-1][i] + 3.*f[j  ][i], 2);
    //         IS1[i] = 13./12. * pow(f[j-1][i] - 2.*f[j  ][i] + f[j+1][i], 2) + 1./4. * pow(   f[j-1][i]               -    f[j+1][i], 2);
    //         IS2[i] = 13./12. * pow(f[j  ][i] - 2.*f[j+1][i] + f[j+2][i], 2) + 1./4. * pow(3.*f[j  ][i] -4.*f[j+1][i] +    f[j+2][i], 2);

    //         // clang-format on
    //     }

    //     auto& alpha0 = IS0;
    //     auto& alpha1 = IS1;
    //     auto& alpha2 = IS2;

    //     for (std::size_t i = 0; i < batch_size; ++i)
    //     {
    //         // (2.16) and Table II (r=3)
    //         alpha0[i] = 0.1 / pow((eps + IS0[i]), 2);
    //         alpha1[i] = 0.6 / pow((eps + IS1[i]), 2);
    //         alpha2[i] = 0.3 / pow((eps + IS2[i]), 2);
    //     }

    //     auto& sum_alphas = flux_values;

    //     // (2.15)
    //     for (std::size_t i = 0; i < batch_size; ++i)
    //     {
    //         sum_alphas[i] = alpha0[i] + alpha1[i] + alpha2[i];
    //     }

    //     // (2.10)
    //     for (std::size_t i = 0; i < batch_size; ++i)
    //     {
    //         flux_values[i] = alpha0[i] / sum_alphas[i] * q0[i] + alpha1[i] / sum_alphas[i] * q1[i] + alpha2[i] / sum_alphas[i] * q2[i];
    //     }
    // }

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
