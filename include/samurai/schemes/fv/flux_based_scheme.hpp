#pragma once
#include "../../interface.hpp"
#include "../explicit_scheme.hpp"

namespace samurai
{

    template <std::size_t output_field_size_, std::size_t stencil_size_, bool is_linear_ = true, bool is_heterogeneous_ = false>
    struct FluxBasedSchemeConfig
    {
        static constexpr std::size_t output_field_size = output_field_size_;
        static constexpr std::size_t stencil_size      = stencil_size_;
        static constexpr bool is_linear                = is_linear_;
        static constexpr bool is_heterogeneous         = is_heterogeneous_;
    };

    /**
     * @class FluxBasedSchemeDefinition
     */
    template <class Field, std::size_t output_field_size, std::size_t stencil_size, bool is_linear_ = true, bool is_heterogeneous_ = false>
    class FluxBasedSchemeDefinition
    {
    };

    /**
     * @class FluxBasedScheme
     */
    template <class DerivedScheme, class cfg, class bdry_cfg, class Field, class check = void>
    class FluxBasedScheme

    {
    };

    /**
     * is_FluxBasedScheme
     */
    template <class Scheme, typename = void>
    struct is_FluxBasedScheme : std::false_type
    {
    };

    template <class Scheme>
    struct is_FluxBasedScheme<
        Scheme,
        std::enable_if_t<
            std::is_base_of_v<FluxBasedScheme<Scheme, typename Scheme::cfg_t, typename Scheme::bdry_cfg_t, typename Scheme::field_t>, Scheme>>>
        : std::true_type
    {
    };

    template <class Scheme>
    inline constexpr bool is_FluxBasedScheme_v = is_FluxBasedScheme<Scheme>::value;

} // end namespace samurai
