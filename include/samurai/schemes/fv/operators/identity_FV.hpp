#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class Field>
    auto make_identity()
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type;

        using cfg      = OneCellStencilFV<SchemeType::LinearHomogeneous, Field::size, Field>;
        using bdry_cfg = BoundaryConfigFV<1>;

        CellBasedScheme<cfg, bdry_cfg> identity;

        identity.set_name("Identity");
        identity.stencil()           = center_only_stencil<dim>();
        identity.coefficients_func() = [](double) -> StencilCoeffs<cfg>
        {
            return {eye<field_value_type, field_size, field_size>()};
        };
        identity.is_symmetric(true);
        identity.is_spd(true);
        return identity;
    }

    template <class Field>
    [[deprecated("Use make_identity() instead.")]] auto make_identity_FV()
    {
        return make_identity<Field>();
    }

} // end namespace samurai
