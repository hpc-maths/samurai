#pragma once
#include "../../interface.hpp"
#include "../explicit_scheme.hpp"
#include "FV_scheme.hpp"
#include "flux_definition.hpp"

namespace samurai
{
    /**
     * @class FluxBasedScheme
     */
    template <class cfg, class bdry_cfg, class check = void>
    class FluxBasedScheme
    {
    };

    template <class cfg>
    auto make_flux_based_scheme(const FluxDefinition<cfg>& flux_definition)
    {
        using bdry_cfg = BoundaryConfigFV<cfg::stencil_size / 2>;

        return FluxBasedScheme<cfg, bdry_cfg>(flux_definition);
    }

    /**
     * is_FluxBasedScheme
     */
    template <class Scheme, typename = void>
    struct is_FluxBasedScheme : std::false_type
    {
    };

    template <class Scheme>
    struct is_FluxBasedScheme<Scheme,
                              std::enable_if_t<std::is_base_of_v<FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::bdry_cfg_t>, Scheme>
                                               || std::is_same_v<FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::bdry_cfg_t>, Scheme>>>
        : std::true_type
    {
    };

    template <class Scheme>
    inline constexpr bool is_FluxBasedScheme_v = is_FluxBasedScheme<Scheme>::value;

} // end namespace samurai
