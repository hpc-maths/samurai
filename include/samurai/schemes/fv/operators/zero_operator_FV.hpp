#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <std::size_t output_field_size, class Field>
    auto make_zero_operator()
    {
        static constexpr std::size_t dim = Field::dim;

        using cfg      = OneCellStencilFV<SchemeType::LinearHomogeneous, output_field_size, Field>;
        using bdry_cfg = BoundaryConfigFV<1>;

        CellBasedScheme<cfg, bdry_cfg> zero;

        using local_matrix_t = typename decltype(zero)::local_matrix_t;

        zero.set_name("Zero");
        zero.stencil()           = center_only_stencil<dim>();
        zero.coefficients_func() = [](double) -> std::array<local_matrix_t, 1>
        {
            return {zeros<local_matrix_t>()};
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
