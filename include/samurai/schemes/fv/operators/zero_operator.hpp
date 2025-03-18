#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <std::size_t output_n_comp, class Field>
    auto make_zero_operator()
    {
        static constexpr std::size_t n_comp = Field::n_comp;
        using field_value_type              = typename Field::value_type;

        using cfg = LocalCellSchemeConfig<SchemeType::LinearHomogeneous, output_n_comp, Field>;

        auto zero = make_cell_based_scheme<cfg>("Zero");

        zero.coefficients_func() = [](double) -> StencilCoeffs<cfg>
        {
            // return {zeros<field_value_type, output_n_comp, n_comp>()};
            StencilCoeffs<cfg> sc;
            sc(0) = zeros<field_value_type, output_n_comp, n_comp>();
            return sc;
        };
        zero.is_symmetric(true);
        return zero;
    }

    template <class Field>
    auto make_zero_operator()
    {
        static constexpr std::size_t default_output_n_comp = Field::n_comp;
        return make_zero_operator<default_output_n_comp, Field>();
    }

    template <class Field>
    [[deprecated("Use make_zero_operator() instead.")]] auto make_zero_operator_FV()
    {
        return make_zero_operator<Field>();
    }

} // end namespace samurai
