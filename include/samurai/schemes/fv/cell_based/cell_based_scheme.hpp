#pragma once
#include "../../explicit_scheme.hpp"
#include "../FV_scheme.hpp"

// #include "flux_definition.hpp"

namespace samurai
{
    enum class SchemeType
    {
        NonLinear,
        LinearHeterogeneous,
        LinearHomogeneous
    };

    /**
     * @class CellBasedScheme
     */
    template <class cfg, class bdry_cfg, class check = void>
    class CellBasedScheme
    {
    };

    template <class cfg>
    auto make_cell_based_scheme(/*const FluxDefinition<cfg>& flux_definition*/)
    {
        using bdry_cfg = BoundaryConfigFV<cfg::stencil_size / 2>;

        return CellBasedScheme<cfg, bdry_cfg>(); // flux_definition);
    }

    /**
     * is_CellBasedScheme
     */
    template <class Scheme, typename = void>
    struct is_CellBasedScheme : std::false_type
    {
    };

    template <class Scheme>
    struct is_CellBasedScheme<Scheme,
                              std::enable_if_t<std::is_base_of_v<CellBasedScheme<typename Scheme::cfg_t, typename Scheme::bdry_cfg_t>, Scheme>>>
        : std::true_type
    {
    };

    template <class Scheme>
    inline constexpr bool is_CellBasedScheme_v = is_CellBasedScheme<Scheme>::value;

} // end namespace samurai
