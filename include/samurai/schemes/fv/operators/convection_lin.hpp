#pragma once
#include "../flux_based/flux_based_scheme__lin_het.hpp"
#include "../flux_based/flux_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <std::size_t dim>
    using VelocityVector = xt::xtensor_fixed<double, xt::xshape<dim>>;

    /**
     * Linear convection, discretized by a (linear) upwind scheme.
     * @param velocity: constant velocity vector
     */
    template <class Field>
    auto make_convection_upwind(const VelocityVector<Field::dim>& velocity)
    {
        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2;

        using cfg = FluxConfig<SchemeType::LinearHomogeneous, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                if (velocity(d) >= 0) // use the left values
                {
                    upwind[d].cons_flux_function = [&](double)
                    {
                        // Return type: 2 matrices (left, right) of size output_field_size x field_size.
                        // In this case, of size field_size x field_size.
                        FluxStencilCoeffs<cfg> coeffs;
                        if constexpr (output_field_size == 1)
                        {
                            coeffs[left]  = velocity(d);
                            coeffs[right] = 0;
                        }
                        else
                        {
                            coeffs[left].fill(0);
                            coeffs[right].fill(0);
                            xt::col(coeffs[left], d)  = velocity(d);
                            xt::col(coeffs[right], d) = 0;
                        }
                        return coeffs;
                    };
                }
                else // use the right values
                {
                    upwind[d].cons_flux_function = [&](double)
                    {
                        FluxStencilCoeffs<cfg> coeffs;
                        if constexpr (output_field_size == 1)
                        {
                            coeffs[left]  = 0;
                            coeffs[right] = velocity(d);
                        }
                        else
                        {
                            coeffs[left].fill(0);
                            coeffs[right].fill(0);
                            xt::col(coeffs[left], d)  = 0;
                            xt::col(coeffs[right], d) = velocity(d);
                        }
                        return coeffs;
                    };
                }
            });

        return make_flux_based_scheme(upwind);
    }

    /**
     * WENO5 implementation.
     * Based on 'Efficent implementation of Weighted ENO schemes', Jiang and Shu, 1996.
     */
    template <class xtensor_t>
    auto compute_weno5_flux(xtensor_t& f)
    {
        static constexpr std::size_t j           = 2; // stencil center in f
        const typename xtensor_t::value_type eps = 1e-6;

        // clang-format off

        // (2.8) and Table I (r=3)
        auto q0 =  1./3 * f(j-2) - 7./6 * f(j-1) + 11./6 * f(j);
        auto q1 = -1./6 * f(j-1) + 5./6 * f(j)   +  1./3 * f(j+1);
        auto q2 =  1./3 * f(j)   + 5./6 * f(j+1) -  1./6 * f(j+2);

        // (3.2)-(3.4)
        auto IS0 = 13./12 * pow(f(j-2) - 2*f(j-1) + f(j)  , 2) + 1./4 * pow(  f(j-2) -4*f(j-1) + 3*f(j),   2);
        auto IS1 = 13./12 * pow(f(j-1) - 2*f(j)   + f(j+1), 2) + 1./4 * pow(  f(j-1)           -   f(j+1), 2);
        auto IS2 = 13./12 * pow(f(j)   - 2*f(j+1) + f(j+2), 2) + 1./4 * pow(3*f(j)   -4*f(j+1) +   f(j+2), 2);

        // clang-format on

        // (2.16) and Table II (r=3)
        auto alpha0 = 0.1 * pow((eps + IS0), -2);
        auto alpha1 = 0.6 * pow((eps + IS1), -2);
        auto alpha2 = 0.3 * pow((eps + IS2), -2);

        // (2.15)
        auto sum_alphas = alpha0 + alpha1 + alpha2;
        auto omega0     = alpha0 / sum_alphas;
        auto omega1     = alpha1 / sum_alphas;
        auto omega2     = alpha2 / sum_alphas;

        // (2.10)
        return omega0 * q0 + omega1 * q1 + omega2 * q2;
    }

    /**
     * Linear convection, discretized by the WENO5 (Jiang & Shu) scheme.
     * @param velocity: constant velocity vector
     */
    template <class Field>
    auto make_convection_weno5(const VelocityVector<Field::dim>& velocity)
    {
        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 6;

        using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

        static_assert(Field::mesh_t::config::ghost_width >= 3, "WENO5 requires at least 3 ghosts.");
        static_assert(dim <= 2, "WENO5 is not implemented for dim > 2.");

        FluxDefinition<cfg> weno5;
        if constexpr (dim == 1)
        {
            // clang-format off
            weno5[0].stencil = {{-2}, {-1}, {0}, {1}, {2}, {3}};
            // clang-format on
        }
        else if constexpr (dim == 2)
        {
            // clang-format off
            weno5[0].direction = {1,0};
            weno5[0].stencil   = {{-2, 0}, {-1, 0}, {0,0}, {1,0}, {2,0}, {3,0}};
            weno5[1].direction = {0,1};
            weno5[1].stencil   = {{ 0,-2}, { 0,-1}, {0,0}, {0,1}, {0,2}, {0,3}};
            // clang-format on
        }

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                // auto flux_f = [&](auto v) -> FluxValue<cfg>
                // {
                //     return velocity(d) * v;
                // };

                if (velocity(d) >= 0)
                {
                    weno5[d].cons_flux_function = [&velocity](auto& cells, const Field& u) -> FluxValue<cfg>
                    {
                        // static constexpr std::size_t stencil_center = 2;

                        xt::xtensor_fixed<double, xt::xshape<5>> f = {u[cells[0]], u[cells[1]], u[cells[2]], u[cells[3]], u[cells[4]]};
                        // f *= vel[cells[stencil_center]](d);
                        f *= velocity(d);

                        return compute_weno5_flux(f);
                    };
                }
                else
                {
                    weno5[d].cons_flux_function = [&velocity](auto& cells, const Field& u) -> FluxValue<cfg>
                    {
                        // static constexpr std::size_t stencil_center = 2;

                        xt::xtensor_fixed<double, xt::xshape<5>> f = {u[cells[5]], u[cells[4]], u[cells[3]], u[cells[2]], u[cells[1]]};
                        // f *= vel[cells[stencil_center]](d);
                        f *= velocity(d);

                        return compute_weno5_flux(f);
                    };
                }
            });

        return make_flux_based_scheme(weno5);
    }

    /**
     * Linear convection, discretized by a (linear) upwind scheme.
     * @param velocity_field: the velocity field
     */
    template <class Field, class VelocityField>
    auto make_convection_upwind(const VelocityField& velocity_field)
    {
        static_assert(Field::dim == VelocityField::dim && VelocityField::size == VelocityField::dim);

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2;

        using cfg = FluxConfig<SchemeType::LinearHeterogeneous, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                upwind[d].cons_flux_function = [&](const auto& cells)
                {
                    // Return type: 2 matrices (left, right) of size output_field_size x field_size.
                    // In this case, of size field_size x field_size.
                    FluxStencilCoeffs<cfg> coeffs;

                    auto velocity = velocity_field[cells[left]];
                    if (velocity(d) >= 0) // use the left values
                    {
                        if constexpr (output_field_size == 1)
                        {
                            coeffs[left]  = velocity(d);
                            coeffs[right] = 0;
                        }
                        else
                        {
                            coeffs[left].fill(0);
                            coeffs[right].fill(0);
                            xt::col(coeffs[left], d)  = velocity(d);
                            xt::col(coeffs[right], d) = 0;
                        }
                    }
                    else // use the right values
                    {
                        if constexpr (output_field_size == 1)
                        {
                            coeffs[left]  = 0;
                            coeffs[right] = velocity(d);
                        }
                        else
                        {
                            coeffs[left].fill(0);
                            coeffs[right].fill(0);
                            xt::col(coeffs[left], d)  = 0;
                            xt::col(coeffs[right], d) = velocity(d);
                        }
                    }
                    return coeffs;
                };
            });

        return make_flux_based_scheme(upwind);
    }

} // end namespace samurai
