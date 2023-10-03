#pragma once
#include "../../explicit_scheme.hpp"
#include "../FV_scheme.hpp"
#include "cell_based_scheme_definition.hpp"

namespace samurai
{
    /**
     * @class CellBasedScheme
     */
    template <class cfg, class bdry_cfg, class check = void>
    class CellBasedScheme
    {
        template <class>
        static constexpr bool dependent_false = false;

        static_assert(
            dependent_false<cfg>,
            "Either the required file has not been included, or the CellBasedScheme class has not been specialized for this type of scheme.");
    };

    template <class cfg, class bdry_cfg>
    auto make_cell_based_scheme()
    {
        return CellBasedScheme<cfg, bdry_cfg>();
    }

    template <class cfg, class bdry_cfg>
    auto make_cell_based_scheme(std::string name)
    {
        auto scheme = make_cell_based_scheme<cfg, bdry_cfg>();
        scheme.set_name(name);
        return scheme;
    }

    template <class cfg>
    auto make_cell_based_scheme()
    {
        using bdry_cfg = BoundaryConfigFV<1>;
        return make_cell_based_scheme<cfg, bdry_cfg>();
    }

    template <class cfg>
    auto make_cell_based_scheme(std::string name)
    {
        auto scheme = make_cell_based_scheme<cfg>();
        scheme.set_name(name);
        return scheme;
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
