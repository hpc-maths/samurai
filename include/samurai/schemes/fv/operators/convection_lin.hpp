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
                    upwind[d].cons_flux_function = [&](double)
                    {
                        // Return type: 2 matrices (left, right) of size output_n_comp x n_comp.
                        // In this case, of size n_comp x n_comp.
                        FluxStencilCoeffs<cfg> coeffs;
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
                        return coeffs;
                    };
                }
                else // use the right values
                {
                    upwind[d].cons_flux_function = [&](double)
                    {
                        FluxStencilCoeffs<cfg> coeffs;
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
                        return coeffs;
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
        requires IsField<VelocityField>
    auto make_convection_upwind(const VelocityField& velocity_field)
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

                upwind[d].cons_flux_function = [&](const auto& cells)
                {
                    // Return type: 2 matrices (left, right) of size output_n_comp x n_comp.
                    // In this case, of size n_comp x n_comp.
                    FluxStencilCoeffs<cfg> coeffs;

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
                    return coeffs;
                };
            });

        auto scheme = make_flux_based_scheme(upwind);
        scheme.set_name("convection");
        scheme.set_parameter_field(velocity_field);
        return scheme;
    }

    /**
     * Linear convection, discretized by a WENO5 (Jiang & Shu) scheme.
     * @param velocity_field: the velocity field
     */
    template <class Field, class VelocityField>
        requires IsField<VelocityField>
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

} // end namespace samurai
