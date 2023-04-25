#pragma once
#include "../cell_based_scheme.hpp"

namespace samurai
{
    namespace petsc
    {
        template <class Field,
                  std::size_t output_field_size,
                  std::size_t dim = Field::dim,
                  class cfg       = OneCellStencilFV<output_field_size>,
                  class bdry_cfg  = BoundaryConfigFV<1>>
        class ZeroOperatorFV : public CellBasedScheme<cfg, bdry_cfg, Field>
        {
            using base_class                = CellBasedScheme<cfg, bdry_cfg, Field>;
            using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

          public:

            using local_matrix_t = typename base_class::local_matrix_t;

            explicit ZeroOperatorFV(Field& unknown)
                : base_class(unknown, center_only_stencil<dim>(), coefficients)
            {
                this->set_name("Zero");
            }

            static std::array<local_matrix_t, cfg::scheme_stencil_size> coefficients(double)
            {
                std::array<local_matrix_t, cfg::scheme_stencil_size> coeffs;
                for (std::size_t i = 0; i < cfg::scheme_stencil_size; i++)
                {
                    coeffs[i] = zeros<local_matrix_t>();
                }
                return coeffs;
            }
        };

        template <std::size_t output_field_size, class Field>
        auto make_zero_operator_FV(Field& f)
        {
            return ZeroOperatorFV<Field, output_field_size>(f);
        }

    } // end namespace petsc
} // end namespace samurai
