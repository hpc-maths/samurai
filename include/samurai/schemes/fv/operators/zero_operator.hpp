#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <std::size_t output_field_size, class Field>
    auto make_zero_operator()
    {
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type;

        using cfg = LocalCellSchemeConfig<SchemeType::LinearHomogeneous, output_field_size, Field>;

        auto zero = make_cell_based_scheme<cfg>("Zero");

        zero.coefficients_func() = [](double) -> StencilCoeffs<cfg>
        {
            // return {zeros<field_value_type, output_field_size, field_size>()};
            StencilCoeffs<cfg> sc;
            sc(0) = zeros<field_value_type, output_field_size, field_size>();
            return sc;
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
