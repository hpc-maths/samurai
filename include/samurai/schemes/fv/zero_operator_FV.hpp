#pragma once
#include "cell_based_scheme.hpp"

namespace samurai
{
    template <class Field, std::size_t output_field_size, class cfg = OneCellStencilFV<output_field_size>, class bdry_cfg = BoundaryConfigFV<1>>
    class ZeroOperatorFV : public CellBasedScheme<cfg, bdry_cfg, Field>
    {
        using base_class = CellBasedScheme<cfg, bdry_cfg, Field>;
        using base_class::dim;
        using local_matrix_t = typename base_class::local_matrix_t;

      public:

        explicit ZeroOperatorFV(Field& unknown)
            : base_class(unknown)
        {
            this->set_name("Zero");
        }

        static constexpr auto stencil()
        {
            return center_only_stencil<dim>();
        }

        static std::array<local_matrix_t, 1> coefficients(double)
        {
            return {zeros<local_matrix_t>()};
        }

        bool matrix_is_symmetric() const override
        {
            return is_uniform(this->unknown().mesh());
        }
    };

    // For some reason this version with an empty stencil is slower...
    /*template <class Field, std::size_t output_field_size, class cfg = EmptyStencilFV<output_field_size>, class bdry_cfg =
    BoundaryConfigFV<1>> class ZeroOperatorFV : public CellBasedScheme<cfg, bdry_cfg, Field>
    {
        using base_class     = CellBasedScheme<cfg, bdry_cfg, Field>;
        using local_matrix_t = typename base_class::local_matrix_t;

      public:

        explicit ZeroOperatorFV(Field& unknown)
            : base_class(unknown, {}, coefficients)
        {
            this->set_name("Zero");
        }

        static std::array<local_matrix_t, 0> coefficients(double)
        {
            return {};
        }
    };*/

    template <std::size_t output_field_size, class Field>
    auto make_zero_operator_FV(Field& f)
    {
        return ZeroOperatorFV<Field, output_field_size>(f);
    }

    template <class Field>
    auto make_zero_operator_FV(Field& f)
    {
        static constexpr std::size_t default_output_field_size = Field::size;
        return make_zero_operator_FV<default_output_field_size>(f);
    }

} // end namespace samurai
