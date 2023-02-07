#pragma once
#include "petsc_cell_based_scheme_assembly.hpp"

namespace samurai { namespace petsc
{
    template<class Field, std::size_t output_field_size, std::size_t dim=Field::dim, class cfg=starStencilFV<dim, output_field_size>>
    class ZeroOperator : public PetscCellBasedSchemeAssembly<cfg, Field>
    {
    public:
        using field_t = Field;
        using local_matrix_t = typename PetscCellBasedSchemeAssembly<cfg, Field>::local_matrix_t;
    public:
        ZeroOperator(Field& unknown) : 
            PetscCellBasedSchemeAssembly<cfg, Field>(unknown, star_stencil<dim>(), coefficients) 
        {}

        bool matrix_is_spd() const override
        {
            return false;
        }

        static std::array<local_matrix_t, cfg::scheme_stencil_size> coefficients(double)
        {
            std::array<local_matrix_t, cfg::scheme_stencil_size> coeffs;
            for (std::size_t i=0; i<cfg::scheme_stencil_size; i++)
                coeffs[i] = zeros<local_matrix_t>();
            return coeffs;
        }
    };

    template<std::size_t output_field_size, class Field>
    auto make_zero_operator(Field& f)
    {
        return ZeroOperator<Field, output_field_size>(f);
    }

}} // end namespace