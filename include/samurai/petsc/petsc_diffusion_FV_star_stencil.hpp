#pragma once
#include "petsc_diffusion_FV.hpp"


// constexpr power function
template <typename T>
constexpr T ce_pow(T num, unsigned int pow)
{
    return pow == 0 ? 1 : num * ce_pow(num, pow-1);
}

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
        // (as defined in FV_stencil())
        1, 

        // ---- Start index and size of contiguous cell indices
        // (as defined in FV_stencil())
        // Here, [left, center, right].
        0, 3
    >;

    /**
     * @class PetscDiffusionFV_StarStencil
     * Assemble the matrix for the problem -Lap(u)=f.
     * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
    */
    template<class Field, std::size_t dim=Field::dim, class cfg=starStencilFV<dim>>
    class PetscDiffusionFV_StarStencil : public PetscDiffusionFV<cfg, Field>
    {
    public:
        using field_t = Field;
        using Mesh = typename Field::mesh_t;
        using mesh_id_t = typename Mesh::mesh_id_t;

        PetscDiffusionFV_StarStencil(Mesh& m) : 
            PetscDiffusionFV<cfg, Field>(m, FV_stencil(), FV_coefficients)
        {}

    private:

        using Stencil = samurai::Stencil<cfg::scheme_stencil_size, dim>;


        /**
         * @return the star stencil of the Finite Volume scheme.
        */
        static constexpr Stencil FV_stencil()
        {
            static_assert(dim >= 1 || dim <= 3, "Finite Volume stencil not implemented for this dimension");

            if constexpr (dim == 1)
            {
                // 3-point stencil:
                //    left, center, right
                return {{-1}, {0}, {1}};
            }
            else if constexpr (dim == 2)
            {
                // 5-point stencil:
                //       left,   center,  right,   bottom,  top 
                return {{-1, 0}, {0, 0},  {1, 0}, {0, -1}, {0, 1}};
            }
            else if constexpr (dim == 3)
            {
                // 7-point stencil:
                //       left,   center,    right,   front,    back,    bottom,    top
                return {{-1,0,0}, {0,0,0},  {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
            }
            return Stencil();
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
         * @brief Enforces Dirichlet boundary condition (on the whole boundary).
         * @param rhs_field rhs_field to update (created by create_rhs()).
         * @param dirichlet Dirichlet function: takes coordinates and returns double.
         *                  It will be called with coordinates on the boundary.
        */
        template<class Func>
        void enforce_dirichlet_bc(Field& field, Func&& dirichlet)
        {
            if (&field.mesh() != &this->mesh)
                assert(false && "Not the same mesh");

            for_each_cell_on_boundary(this->mesh, FV_stencil(), FV_coefficients,
            [&] (const auto& cell, const auto& towards_bdry, double out_coeff)
            {
                auto boundary_point = cell.face_center(towards_bdry);
                auto dirichlet_value = dirichlet(boundary_point);
                field[cell] -= 2 * out_coeff * dirichlet_value;
            });
        }

        /**
         * @brief Creates a coarse object from a coarse mesh and a fine object.
         * @note  This method is used by the multigrid.
        */
        static PetscDiffusionFV_StarStencil create_coarse(const PetscDiffusionFV_StarStencil& /*fine*/, Mesh& coarse_mesh)
        {
            return PetscDiffusionFV_StarStencil(coarse_mesh);
        }
    };

}} // end namespace