#pragma once
#include "../flux_based/flux_based_scheme__lin_het.hpp"
#include "../flux_based/flux_based_scheme__lin_hom.hpp"
#include "weno_impl.hpp"

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
        static constexpr std::size_t dim = Field::dim;

        static constexpr std::size_t stencil_size = 2;
        using input_field_t                       = Field;
        using output_field_t                      = Field;

        using cfg = FluxConfig<SchemeType::LinearHomogeneous, stencil_size, output_field_t, input_field_t>;

        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                if (velocity(d) >= 0) // use the left values
                {
                    upwind[d].cons_flux_function = [&](FluxStencilCoeffs<cfg>& coeffs, double)
                    {
                        // Return type: 2 matrices (left, right) of size output_n_comp x n_comp.
                        // In this case, of size n_comp x n_comp.
                        if constexpr (Field::is_scalar)
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
                    };
                }
                else // use the right values
                {
                    upwind[d].cons_flux_function = [&](FluxStencilCoeffs<cfg>& coeffs, double)
                    {
                        if constexpr (Field::is_scalar)
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
                    };
                }
            });

        auto scheme = make_flux_based_scheme(upwind);
        scheme.set_name("convection");
        return scheme;
    }

    /**
     * Linear convection, discretized by the WENO5 (Jiang & Shu) scheme.
     * @param velocity: constant velocity vector
     */
    template <class Field>
    auto make_convection_weno5(const VelocityVector<Field::dim>& velocity)
    {
        static constexpr std::size_t dim          = Field::dim;
        static constexpr bool is_soa              = detail::is_soa_v<Field>;
        static constexpr std::size_t stencil_size = 6;
        using input_field_t                       = Field;
        using output_field_t                      = Field;

        using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, output_field_t, input_field_t>;

        FluxDefinition<cfg> weno5;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                // Stencil creation:
                //        weno5[0].stencil = {{-2, 0}, {-1, 0}, {0,0}, {1,0}, {2,0}, {3,0}};
                //        weno5[1].stencil = {{ 0,-2}, { 0,-1}, {0,0}, {0,1}, {0,2}, {0,3}};
                weno5[d].stencil = line_stencil<dim, d>(-2, -1, 0, 1, 2, 3);

                if (velocity(d) >= 0)
                {
                    weno5[d].cons_flux_function =
                        [&velocity](FluxValue<cfg>& flux, const StencilData<cfg>& /*data*/, const StencilValues<cfg>& u)
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[0], u[1], u[2], u[3], u[4]});
                        f *= velocity(d);
                        compute_weno5_flux(flux, f);
                    };
                    weno5[d].cons_jacobian_function =
                        [&velocity](StencilJacobian<cfg>& jac, const StencilData<cfg>& /*data*/, const StencilValues<cfg>& u)
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[0], u[1], u[2], u[3], u[4]});
                        f *= velocity(d);

                        std::array<decltype(&jac[0]), 5> jacobians({&jac[0], &jac[1], &jac[2], &jac[3], &jac[4]});
                        if constexpr (Field::is_scalar)
                        {
                            jac[5] = 0; // the last one is not used
                        }
                        else
                        {
                            jac[5].fill(0); // the last one is not used
                        }
                        compute_weno5_jacobian(jacobians, f);
                        jac *= velocity(d);
                    };
                }
                else
                {
                    weno5[d].cons_flux_function =
                        [&velocity](FluxValue<cfg>& flux, const StencilData<cfg>& /*data*/, const StencilValues<cfg>& u)
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[5], u[4], u[3], u[2], u[1]});
                        f *= velocity(d);
                        compute_weno5_flux(flux, f);
                    };
                    weno5[d].cons_jacobian_function =
                        [&velocity](StencilJacobian<cfg>& jac, const StencilData<cfg>& /*data*/, const StencilValues<cfg>& u)
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[5], u[4], u[3], u[2], u[1]});
                        f *= velocity(d);

                        std::array<decltype(&jac[0]), 5> jacobians({&jac[5], &jac[4], &jac[3], &jac[2], &jac[1]});
                        if constexpr (Field::is_scalar)
                        {
                            jac[0] = 0; // the first one is not used
                        }
                        else
                        {
                            jac[0].fill(0); // the first one is not used
                        }
                        compute_weno5_jacobian(jacobians, f);
                        jac *= velocity(d);
                    };
                }
            });

        auto scheme = make_flux_based_scheme(weno5);
        scheme.set_name("convection");
        return scheme;
    }

    /**
     * Linear convection, discretized by a (linear) upwind scheme.
     * @param velocity_field: the velocity field
     */
    template <class Field, class VelocityField>
        requires field_like<VelocityField>
    auto make_convection_upwind(VelocityField& velocity_field)
    {
        static_assert(Field::dim == VelocityField::dim && VelocityField::n_comp == VelocityField::dim);

        static constexpr std::size_t dim    = Field::dim;
        static constexpr std::size_t n_comp = Field::n_comp;

        static constexpr std::size_t stencil_size = 2;
        using input_field_t                       = Field;
        using output_field_t                      = Field;
        using parameter_field_t                   = VelocityField;

        using cfg = FluxConfig<SchemeType::LinearHeterogeneous, stencil_size, output_field_t, input_field_t, parameter_field_t>;

        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                upwind[d].cons_flux_function = [&](FluxStencilCoeffs<cfg>& coeffs, const StencilData<cfg>& data)
                {
                    // Return type: 2 matrices (left, right) of size output_n_comp x n_comp.
                    // In this case, of size n_comp x n_comp.

                    const auto& cells = data.cells;

                    auto velocity = 0.5 * (velocity_field[cells[left]] + velocity_field[cells[right]]);
                    if (velocity(d) >= 0) // use the left values
                    {
                        if constexpr (n_comp == 1)
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
                        if constexpr (n_comp == 1)
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
                };
            });

        auto scheme = make_flux_based_scheme(upwind);
        scheme.set_name("convection");
        scheme.set_parameter_field(velocity_field);
        return scheme;
    }

    /*template <class TemperatureField, class VelocityField>
    auto make_dual_convection_upwind(TemperatureField& T)
    {
        static_assert(TemperatureField::dim == VelocityField::dim && VelocityField::n_comp == VelocityField::dim);

        static constexpr std::size_t dim    = VelocityField::dim;
        static constexpr std::size_t n_comp = VelocityField::n_comp;

        static constexpr std::size_t stencil_size = 2;
        using input_field_t                       = VelocityField;
        using output_field_t                      = TemperatureField;
        using parameter_field_t                   = TemperatureField;

        using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, output_field_t, input_field_t, parameter_field_t>;

        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                upwind[d].cons_jacobian_function =
                    [&](StencilJacobian<cfg>& jac, const StencilData<cfg>& data, const StencilValues<cfg>& velocity_data)
                {
                    const auto& cells = data.cells;
                    auto velocity     = 0.5 * (velocity_data[left] + velocity_data[right]);
                    if (velocity(d) >= 0) // use the left values
                    {
                        if constexpr (n_comp == 1)
                        {
                            jac[left]  = 0.5 * T[cells[left]];
                            jac[right] = 0.5 * T[cells[left]];
                        }
                        else
                        {
                            jac[left].fill(0);
                            jac[right].fill(0);
                            xt::col(jac[left], d)  = 0.5 * T[cells[left]];
                            xt::col(jac[right], d) = 0.5 * T[cells[left]];
                        }
                    }
                    else // use the right values
                    {
                        if constexpr (n_comp == 1)
                        {
                            jac[left]  = 0.5 * T[cells[right]];
                            jac[right] = 0.5 * T[cells[right]];
                        }
                        else
                        {
                            jac[left].fill(0);
                            jac[right].fill(0);
                            xt::col(jac[left], d)  = 0.5 * T[cells[right]];
                            xt::col(jac[right], d) = 0.5 * T[cells[right]];
                        }
                    }
                };
            });

        auto scheme = make_flux_based_scheme(upwind);
        scheme.set_name("dual_convection");
        scheme.set_parameter_field(T);
        return scheme;
    }*/

    /**
     * Linear convection, discretized by a WENO5 (Jiang & Shu) scheme.
     * @param velocity_field: the velocity field
     */
    template <class Field, class VelocityField>
        requires field_like<VelocityField>
    auto make_convection_weno5(VelocityField& velocity_field)
    {
        static constexpr std::size_t dim = Field::dim;
        static constexpr bool is_soa     = detail::is_soa_v<Field>;

        static constexpr std::size_t stencil_size = 6;
        using input_field_t                       = Field;
        using output_field_t                      = Field;
        using parameter_field_t                   = VelocityField;

        using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, output_field_t, input_field_t, parameter_field_t>;

        FluxDefinition<cfg> weno5;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto d_)
            {
                static constexpr std::size_t d = d_();

                // Stencil creation:
                //        weno5[0].stencil = {{-2, 0}, {-1, 0}, {0,0}, {1,0}, {2,0}, {3,0}};
                //        weno5[1].stencil = {{ 0,-2}, { 0,-1}, {0,0}, {0,1}, {0,2}, {0,3}};
                weno5[d].stencil = line_stencil<dim, d>(-2, -1, 0, 1, 2, 3);

                weno5[d].cons_flux_function =
                    [&velocity_field](FluxValue<cfg>& flux, const StencilData<cfg>& data, const StencilValues<cfg>& u)
                {
                    static constexpr std::size_t left  = 2;
                    static constexpr std::size_t right = 3;

                    auto v = 0.5 * (velocity_field[data.cells[left]](d) + velocity_field[data.cells[right]](d));

                    if (v >= 0)
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[0], u[1], u[2], u[3], u[4]});
                        f *= v;
                        compute_weno5_flux(flux, f);
                    }
                    else
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[5], u[4], u[3], u[2], u[1]});
                        f *= v;
                        compute_weno5_flux(flux, f);
                    }
                };
            });

        auto scheme = make_flux_based_scheme(weno5);
        scheme.set_name("convection");
        scheme.set_parameter_field(velocity_field);
        return scheme;
    }

    template <class Field, class VelocityField>
        requires field_like<VelocityField>
    auto make_convection_smooth_rusanov_incompressible(VelocityField& velocity_field)
    {
        static constexpr std::size_t dim          = Field::dim;
        static constexpr std::size_t stencil_size = 2;

        constexpr std::size_t left  = 0;
        constexpr std::size_t right = 1;

        using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

        FluxDefinition<cfg> smooth_rusanov;

        auto smooth_abs = [](auto x)
        {
            constexpr double eps = 1e-6;
            return std::sqrt(x * x + eps * eps);
        };

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                smooth_rusanov[d].cons_flux_function =
                    [smooth_abs, &velocity_field](FluxValue<cfg>& flux, const StencilData<cfg>& data, const StencilValues<cfg>& u)
                {
                    const auto& uL = u[left];
                    const auto& uR = u[right];

                    auto vL = velocity_field[data.cells[left]](d);
                    auto vR = velocity_field[data.cells[right]](d);

                    // The textbook formula is λ = max(|vL|, |vR|).
                    // In the incompressible case, there are no shocks, so vL ≈ (vL+vR)/2 ≈ vR.
                    // Consequently, λ ≈ |(vL+vR)/2|.
                    // Finally, to avoid non-differentiability at 0, we use a smooth approximation of the absolute value.
                    auto lambda = smooth_abs(0.5 * (vL + vR));

                    auto F_L = vL * uL;
                    auto F_R = vR * uR;

                    flux = 0.5 * (F_L + F_R - lambda * (uR - uL));
                };

                smooth_rusanov[d].cons_jacobian_function =
                    [smooth_abs, &velocity_field](StencilJacobian<cfg>& jac, const StencilData<cfg>& data, const StencilValues<cfg>& /*u*/)
                {
                    auto vL = velocity_field[data.cells[left]](d);
                    auto vR = velocity_field[data.cells[right]](d);

                    auto lambda = smooth_abs(0.5 * (vL + vR));

                    if constexpr (Field::is_scalar)
                    {
                        jac[left]  = 0.5 * (vL + lambda);
                        jac[right] = 0.5 * (vR - lambda);
                    }
                    else
                    {
                        jac[left].fill(0);
                        jac[right].fill(0);
                        for (std::size_t i = 0; i < Field::n_comp; ++i)
                        {
                            jac[left](i, i)  = 0.5 * (vL + lambda);
                            jac[right](i, i) = 0.5 * (vR - lambda);
                        }
                    }
                };
            });
        auto scheme = make_flux_based_scheme(smooth_rusanov);
        scheme.set_name("smooth rusanov");
        return scheme;
    }

    template <class Field, class VelocityField>
    auto make_dual_convection_smooth_rusanov_incompressible(Field& u)
    {
        static_assert(Field::dim == VelocityField::dim && VelocityField::n_comp == VelocityField::dim);
        static_assert(Field::is_scalar, "The field in parameter must be scalar");

        static constexpr std::size_t dim          = VelocityField::dim;
        static constexpr std::size_t stencil_size = 2;
        using input_field_t                       = VelocityField;
        using output_field_t                      = Field;
        using parameter_field_t                   = Field;

        using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, output_field_t, input_field_t, parameter_field_t>;

        FluxDefinition<cfg> dual_rusanov;

        auto diff_smooth_abs = [](auto x)
        {
            constexpr double eps = 1e-6;
            return x / std::sqrt(x * x + eps * eps);
        };

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                dual_rusanov[d].cons_jacobian_function =
                    [&](StencilJacobian<cfg>& jac, const StencilData<cfg>& data, const StencilValues<cfg>& velocity_data)
                {
                    auto vL = velocity_data[left](d);
                    auto vR = velocity_data[right](d);

                    auto uL = u[data.cells[left]];
                    auto uR = u[data.cells[right]];

                    jac[left].fill(0);
                    jac[right].fill(0);

                    // Differentiate lambda = smooth_abs(0.5 * (vL + vR)) w.r.t. vL and vR
                    auto dlambda_dvL = 0.5 * diff_smooth_abs(0.5 * (vL + vR));
                    auto dlambda_dvR = dlambda_dvL;

                    // Differentiate flux = 0.5 * (vL*uL + vR*uR - lambda*(uR - uL)) w.r.t. vL and vR
                    xt::col(jac[left], d)  = 0.5 * (uL - dlambda_dvL * (uR - uL));
                    xt::col(jac[right], d) = 0.5 * (uR - dlambda_dvR * (uR - uL));
                };
            });

        auto scheme = make_flux_based_scheme(dual_rusanov);
        scheme.set_name("dual_convection");
        scheme.set_parameter_field(u);
        return scheme;
    }

} // end namespace samurai
