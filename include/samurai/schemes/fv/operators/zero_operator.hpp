#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class output_field_t, class input_field_t>
    auto make_zero_operator()
    {
        using field_value_type = typename input_field_t::value_type;

        using cfg = LocalCellSchemeConfig<SchemeType::LinearHomogeneous, output_field_t, input_field_t>;

        auto zero = make_cell_based_scheme<cfg>("Zero");

        zero.coefficients_func() = [](double) -> StencilCoeffs<cfg>
        {
            StencilCoeffs<cfg> sc;
            sc(0) = zeros<field_value_type, output_field_t::n_comp, input_field_t::n_comp, input_field_t::is_scalar>();
            return sc;
        };
        zero.is_symmetric(true);
        return zero;
    }

    template <class Field>
    auto make_zero_operator()
    {
        using default_output_field_t = Field;
        return make_zero_operator<default_output_field_t, Field>();
    }

} // end namespace samurai
