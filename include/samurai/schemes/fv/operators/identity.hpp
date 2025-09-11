#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class Field>
    auto make_identity()
    {
        static constexpr std::size_t n_comp = Field::n_comp;
        using field_value_type              = typename Field::value_type;

        using cfg = LocalCellSchemeConfig<SchemeType::LinearHomogeneous, Field, Field>;

        auto identity = make_cell_based_scheme<cfg>("Identity");

        identity.coefficients_func() = [](StencilCoeffs<cfg>& sc, double)
        {
            sc = eye<field_value_type, n_comp, n_comp, Field::is_scalar>();
        };
        identity.is_symmetric(true);
        identity.is_spd(true);
        return identity;
    }

} // end namespace samurai
