#pragma once
#include "petsc_cell_based_scheme_assembly.hpp"

namespace samurai { namespace petsc
{
    /**
     * @class PetscDiffusionFV_StarStencil
     * Assemble the matrix for the problem -Lap(u)=f.
     * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
    */
    template<class Field, std::size_t dim=Field::dim, class cfg=starStencilFV<dim, DirichletEnforcement::Elimination>>
    class PetscDiffusionFV_StarStencil : public PetscCellBasedSchemeAssembly<cfg, Field>
    {
    public:
        using field_t = Field;
        using Mesh = typename Field::mesh_t;
        using boundary_condition_t = typename Field::boundary_condition_t;

        PetscDiffusionFV_StarStencil(Mesh& m, const std::vector<boundary_condition_t>& boundary_conditions) : 
            PetscCellBasedSchemeAssembly<cfg, Field>(m, star_stencil<dim>(), coefficients, boundary_conditions)
        {}

        bool matrix_is_spd() override
        {
            // The projections/predictions kill the symmetry, so the matrix is spd only if the mesh is not refined.
            return this->mesh.min_level() == this->mesh.max_level();
        }

        static std::array<double, cfg::scheme_stencil_size> coefficients(double h)
        {
            double one_over_h2 = 1/(h*h);

            std::array<double, cfg::scheme_stencil_size> coeffs;
            for (unsigned int i = 0; i<cfg::scheme_stencil_size; ++i)
            {
                coeffs[i] = -one_over_h2;
            }
            coeffs[cfg::center_index] = (cfg::scheme_stencil_size-1) * one_over_h2;
            return coeffs;
        }

        /**
         * @brief Creates a coarse object from a coarse mesh and a fine object.
         * @note  This method is used by the multigrid.
        */
        static PetscDiffusionFV_StarStencil create_coarse(const PetscDiffusionFV_StarStencil& fine, Mesh& coarse_mesh)
        {
            return PetscDiffusionFV_StarStencil(coarse_mesh, fine._boundary_conditions);
        }
    };

}} // end namespace