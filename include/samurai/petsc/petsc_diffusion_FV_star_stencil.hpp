#pragma once
#include "petsc_cell_based_scheme_assembly.hpp"

namespace samurai { namespace petsc
{
    /**
     * Set useful sizes to define the sparsity pattern of the matrix and perform the preallocation.
    */
    template<std::size_t dim>
    using starStencilFV = PetscAssemblyConfig
    <
        // ----  Stencil size 
        // Cell-centered Finite Volume scheme:
        // center + 1 neighbour in each Cartesian direction (2*dim directions) --> 1+2=3 in 1D
        //                                                                         1+4=5 in 2D
        1 + 2*dim,

        // ----  Projection stencil size
        // cell + 2^dim children --> 1+2=3 in 1D 
        //                           1+4=5 in 2D
        1 + (1 << dim), 

        // ----  Prediction stencil size
        // Here, order 1:
        // cell + hypercube of 3 cells --> 1+3= 4 in 1D
        //                                 1+9=10 in 2D
        1 + ce_pow(3, dim), 

        // ---- Index of the stencil center
        // (as defined in star_stencil())
        1, 

        // ---- Start index and size of contiguous cell indices
        // (as defined in star_stencil())
        // Here, [left, center, right].
        0, 3
    >;

    /**
     * @class PetscDiffusionFV_StarStencil
     * Assemble the matrix for the problem -Lap(u)=f.
     * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
    */
    template<class Field, std::size_t dim=Field::dim, class cfg=starStencilFV<dim>>
    class PetscDiffusionFV_StarStencil : public PetscCellBasedSchemeAssembly<cfg, Field>
    {
    public:
        using field_t = Field;
        using Mesh = typename Field::mesh_t;
        using boundary_condition_t = typename Field::boundary_condition_t;

        PetscDiffusionFV_StarStencil(Mesh& m, const std::vector<boundary_condition_t>& boundary_conditions) : 
            PetscCellBasedSchemeAssembly<cfg, Field>(m, star_stencil<dim>(), FV_coefficients, boundary_conditions)
        {}

    private:

        bool matrix_is_spd() override
        {
            // The projections/predictions kill the symmetry, so the matrix is spd only if the mesh is not refined.
            return this->mesh.min_level() == this->mesh.max_level();
        }

        static std::array<double, cfg::scheme_stencil_size> FV_coefficients(double h)
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

    public:
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