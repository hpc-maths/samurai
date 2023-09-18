#pragma once
#include "../cell_based_scheme.hpp"

namespace samurai
{
    template <class Field,
              std::size_t output_field_size,
              // scheme config
              class cfg      = OneCellStencilFV<output_field_size>,
              class bdry_cfg = BoundaryConfigFV<1>>
    class ZeroOperatorFV : public CellBasedScheme<cfg, bdry_cfg, Field>
    {
        using base_class = CellBasedScheme<cfg, bdry_cfg, Field>;
        using base_class::dim;
        using local_matrix_t = typename base_class::local_matrix_t;

      public:

        ZeroOperatorFV()
        {
            this->set_name("Zero");
            this->stencil()           = center_only_stencil<dim>();
            this->coefficients_func() = [](double) -> std::array<local_matrix_t, 1>
            {
                return {zeros<local_matrix_t>()};
            };
            this->is_symmetric(true);
        }
    };

    template <std::size_t output_field_size, class Field>
    auto make_zero_operator()
    {
        return ZeroOperatorFV<Field, output_field_size>();
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
