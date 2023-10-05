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
                static constexpr int d = decltype(integral_constant_d)::value;

                multiplied_scheme.flux_definition()[d] = scheme.flux_definition()[d];
                if (scalar != 1)
                {
                    // Multiply the flux function by the scalar
                    if constexpr (cfg::scheme_type == SchemeType::LinearHomogeneous)
                    {
                        multiplied_scheme.flux_definition()[d].flux_function = [=](auto h)
                        {
                            return scalar * scheme.flux_definition()[d].flux_function(h);
                        };
                    }
                    else if constexpr (cfg::scheme_type == SchemeType::LinearHeterogeneous)
                    {
                        multiplied_scheme.flux_definition()[d].flux_function = [=](auto& cells)
                        {
                            return scalar * scheme.flux_definition()[d].flux_function(cells);
                        };
                    }
                    else // SchemeType::NonLinear
                    {
                        multiplied_scheme.flux_definition()[d].flux_function = [=](auto& cells, auto& field)
                        {
                            return scalar * scheme.flux_definition()[d].flux_function(cells, field);
                        };
                    }
                }
            });

        multiplied_scheme.is_spd(scheme.is_spd() && scalar != 0);
        multiplied_scheme.set_name(std::to_string(scalar) + " * " + scheme.name());
        return multiplied_scheme;
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
