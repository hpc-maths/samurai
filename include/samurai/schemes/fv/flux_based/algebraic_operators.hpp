#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
    template <class cfg, class bdry_cfg>
    auto operator*(double scalar, const FluxBasedScheme<cfg, bdry_cfg>& scheme)
    {
        static constexpr std::size_t dim = cfg::dim;

        FluxBasedScheme<cfg, bdry_cfg> multiplied_scheme(scheme); // copy

        static_for<0, dim>::apply(
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                multiplied_scheme.flux_definition()[d] = scheme.flux_definition()[d];
                if (scalar != 1)
                {
                    // Multiply the flux function by the scalar
                    if constexpr (cfg::scheme_type == SchemeType::LinearHomogeneous)
                    {
                        multiplied_scheme.flux_definition()[d].cons_flux_function = [=](auto h) -> FluxStencilCoeffs<cfg>
                        {
                            return scalar * scheme.flux_definition()[d].cons_flux_function(h);
                        };
                    }
                    else if constexpr (cfg::scheme_type == SchemeType::LinearHeterogeneous)
                    {
                        multiplied_scheme.flux_definition()[d].cons_flux_function = [=](auto& cells) -> FluxStencilCoeffs<cfg>
                        {
                            return scalar * scheme.flux_definition()[d].cons_flux_function(cells);
                        };
                    }
                    else // SchemeType::NonLinear
                    {
                        if (scheme.flux_definition()[d].cons_flux_function)
                        {
                            multiplied_scheme.flux_definition()[d].cons_flux_function = [=](auto& cells, const auto& field) -> FluxValue<cfg>
                            {
                                return scalar * scheme.flux_definition()[d].cons_flux_function(cells, field);
                            };
                        }
                        if (scheme.flux_definition()[d].flux_function)
                        {
                            multiplied_scheme.flux_definition()[d].flux_function = [=](auto& cells, const auto& field) -> FluxValuePair<cfg>
                            {
                                return scalar * scheme.flux_definition()[d].flux_function(cells, field);
                            };
                        }
                        if (scheme.flux_definition()[d].cons_jacobian_function)
                        {
                            multiplied_scheme.flux_definition()[d].cons_jacobian_function = [=](auto& cells,
                                                                                                const auto& field) -> StencilJacobian<cfg>
                            {
                                return scalar * scheme.flux_definition()[d].cons_jacobian_function(cells, field);
                            };
                        }
                        if (scheme.flux_definition()[d].jacobian_function)
                        {
                            multiplied_scheme.flux_definition()[d].jacobian_function = [=](auto& cells,
                                                                                           const auto& field) -> StencilJacobianPair<cfg>
                            {
                                return scalar * scheme.flux_definition()[d].jacobian_function(cells, field);
                            };
                        }
                    }
                }
            });

        multiplied_scheme.is_spd(scheme.is_spd() && scalar != 0);
        multiplied_scheme.set_name(std::to_string(scalar) + " * " + scheme.name());
        return multiplied_scheme;
    }

    /**
     * Binary '+' operator if same config
     */
    template <class cfg, class bdry_cfg>
    FluxBasedScheme<cfg, bdry_cfg> operator+(const FluxBasedScheme<cfg, bdry_cfg>& scheme1, const FluxBasedScheme<cfg, bdry_cfg>& scheme2)
    {
        FluxBasedScheme<cfg, bdry_cfg> sum_scheme(scheme1); // copy
        sum_scheme.set_name(scheme1.name() + " + " + scheme2.name());

        static_for<0, cfg::dim>::apply(
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                if constexpr (cfg::scheme_type == SchemeType::LinearHomogeneous)
                {
                    sum_scheme.flux_definition()[d].cons_flux_function = [=](auto h)
                    {
                        return scheme1.flux_definition()[d].cons_flux_function(h) + scheme2.flux_definition()[d].cons_flux_function(h);
                    };
                }
                else if constexpr (cfg::scheme_type == SchemeType::LinearHeterogeneous)
                {
                    sum_scheme.flux_definition()[d].cons_flux_function = [=](auto& cells)
                    {
                        return scheme1.flux_definition()[d].cons_flux_function(cells)
                             + scheme2.flux_definition()[d].cons_flux_function(cells);
                    };
                }
                else // SchemeType::NonLinear
                {
                    if (scheme1.flux_definition()[d].flux_function && scheme2.flux_definition()[d].flux_function)
                    {
                        sum_scheme.flux_definition()[d].flux_function = [=](auto& cells, const auto& field)
                        {
                            return scheme1.flux_definition()[d].flux_function(cells, field)
                                 + scheme2.flux_definition()[d].flux_function(cells, field);
                        };
                        sum_scheme.cons_flux_function = nullptr;
                    }
                    else if (scheme1.flux_definition()[d].cons_flux_function && scheme2.flux_definition()[d].cons_flux_function)
                    {
                        sum_scheme.flux_definition()[d].cons_flux_function = [=](auto& cells, const auto& field)
                        {
                            return scheme1.flux_definition()[d].cons_flux_function(cells, field)
                                 + scheme2.flux_definition()[d].cons_flux_function(cells, field);
                        };
                        sum_scheme.flux_function = nullptr;
                    }
                    else
                    {
                        assert(false && "The case where scheme1.flux_function and scheme2.cons_flux_function are set is not implemented.");
                    }

                    if (scheme1.flux_definition()[d].jacobian_function && scheme2.flux_definition()[d].jacobian_function)
                    {
                        sum_scheme.flux_definition()[d].jacobian_function = [=](auto& cells, const auto& field)
                        {
                            return scheme1.flux_definition()[d].jacobian_function(cells, field)
                                 + scheme2.flux_definition()[d].jacobian_function(cells, field);
                        };
                        sum_scheme.cons_jacobian_function = nullptr;
                    }
                    else if (scheme1.flux_definition()[d].cons_jacobian_function && scheme2.flux_definition()[d].cons_jacobian_function)
                    {
                        sum_scheme.flux_definition()[d].cons_jacobian_function = [=](auto& cells, const auto& field)
                        {
                            return scheme1.flux_definition()[d].cons_jacobian_function(cells, field)
                                 + scheme2.flux_definition()[d].cons_jacobian_function(cells, field);
                        };
                        sum_scheme.flux_function = nullptr;
                    }
                    else
                    {
                        assert(false
                               && "The case where scheme1.jacobian_function and scheme2.cons_jacobian_function are set is not implemented.");
                    }
                }
            });
        return sum_scheme;
    }

    template <class cfg, class bdry_cfg>
    auto operator-(const FluxBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return (-1) * scheme;
    }

    template <class cfg, class bdry_cfg>
    auto operator-(FluxBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return (-1) * scheme;
    }

    template <class cfg, class bdry_cfg>
    auto operator-(FluxBasedScheme<cfg, bdry_cfg>&& scheme)
    {
        return (-1) * scheme;
    }

} // end namespace samurai
