#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <std::size_t output_field_size, class Field>
    auto make_zero_operator()
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type;

        using cfg      = OneCellStencilFV<SchemeType::LinearHomogeneous, output_field_size, Field>;
        using bdry_cfg = BoundaryConfigFV<1>;

        CellBasedScheme<cfg, bdry_cfg> zero;

        zero.set_name("Zero");
        zero.stencil()           = center_only_stencil<dim>();
        zero.coefficients_func() = [](double) -> StencilCoeffs<cfg>
        {
            return {zeros<field_value_type, output_field_size, field_size>()};
        };
        zero.is_symmetric(true);
        return zero;
    }

    template <class Field>
    auto make_zero_operator()
    {
        static constexpr std::size_t default_output_field_size = Field::size;
        return make_zero_operator<default_output_field_size, Field>();
    }

    template <class Field>
    [[deprecated("Use make_zero_operator() instead.")]] auto make_zero_operator_FV()
    {
        return make_zero_operator<Field>();
    }

} // end namespace samurai
