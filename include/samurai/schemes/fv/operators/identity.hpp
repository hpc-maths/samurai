#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class Field>
    auto make_identity()
    {
        static constexpr std::size_t n_comp = Field::n_comp;
        using field_value_type              = typename Field::value_type;

        using cfg = LocalCellSchemeConfig<SchemeType::LinearHomogeneous, Field::n_comp, Field>;

        auto identity = make_cell_based_scheme<cfg>("Identity");

        identity.coefficients_func() = [](double) -> StencilCoeffs<cfg>
        {
            // return {eye<field_value_type, n_comp, n_comp>()};
            StencilCoeffs<cfg> sc;
            sc(0) = eye<field_value_type, n_comp, n_comp>();
            return sc;
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
