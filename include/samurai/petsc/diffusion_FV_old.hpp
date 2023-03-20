#pragma once
#include "cell_based_scheme.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * @class DiffusionFV
         * Assemble the matrix for the problem -Lap(u)=f.
         * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
        */
        template<class Field, DirichletEnforcement dirichlet_enfcmt=Equation, std::size_t dim=Field::dim, class cfg=StarStencilFV<dim, Field::size, 1, dirichlet_enfcmt>>
        class DiffusionFV_old : public CellBasedScheme<cfg, Field>
        {
          public:

            using field_t              = Field;
            using Mesh                 = typename Field::mesh_t;
            using local_matrix_t       = typename CellBasedScheme<cfg, Field>::local_matrix_t;
            using boundary_condition_t = typename Field::boundary_condition_t;

            DiffusionFV_old(Field& unknown) : 
                CellBasedScheme<cfg, Field>(unknown, stencil(), coefficients)
            {}

            static constexpr auto stencil()
            {
                return star_stencil<dim>();
            }

            bool matrix_is_spd() const override
            {
                if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                {
                    // The projections/predictions kill the symmetry, so the
                    // matrix is spd only if the mesh is uniform.
                    return this->mesh().min_level() == this->mesh().max_level();
                }
                else
                {
                    return false;
                }
            }

            static std::array<local_matrix_t, cfg::scheme_stencil_size> coefficients(double h)
            {
                double one_over_h2 = 1 / (h * h);
                auto Identity      = eye<local_matrix_t>();
                std::array<local_matrix_t, cfg::scheme_stencil_size> coeffs;
                for (unsigned int i = 0; i < cfg::scheme_stencil_size; ++i)
                {
                    coeffs[i] = -one_over_h2 * Identity;
                }
                coeffs[cfg::center_index] = (cfg::scheme_stencil_size - 1) * one_over_h2 * Identity;
                return coeffs;
            }

            /**
             * @brief Creates a coarse object from a coarse mesh and a fine
             * object.
             * @note  This method is used by the multigrid.
            */
            static DiffusionFV_old create_coarse(const DiffusionFV_old& fine, Mesh& coarse_mesh)
            {
                return DiffusionFV_old(coarse_mesh, fine.m_boundary_conditions);
            }
        };


        template<DirichletEnforcement dirichlet_enfcmt=Equation, class Field>
        auto make_diffusion_FV_old(Field& f)
        {
            return DiffusionFV_old<Field, dirichlet_enfcmt>(f);
        }

    } // end namespace petsc
} // end namespace samurai