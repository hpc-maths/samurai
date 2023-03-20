#pragma once
#include "flux_based_scheme.hpp"

namespace samurai 
{ 
    namespace petsc
    {
        /**
         * @class DiffusionFV
         * Assemble the matrix for the problem -Lap(u)=f.
         * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
        */
        template<class Field, DirichletEnforcement dirichlet_enfcmt=Equation, std::size_t dim=Field::dim, class cfg=FluxBasedAssemblyConfig<Field::size, 2, dirichlet_enfcmt>>
        class DiffusionFV : public FluxBasedScheme<cfg, Field>
        {
        public:
            using field_t = Field;
            using Mesh = typename Field::mesh_t;
            using flux_computation_t = typename FluxBasedScheme<cfg, Field>::flux_computation_t;
            using coeff_matrix_t = typename flux_computation_t::coeff_matrix_t;

            DiffusionFV(Field& unknown) : 
                FluxBasedScheme<cfg, Field>(unknown, fluxes())
            {}

            static auto fluxes()
            {
                std::array<flux_computation_t, dim> fluxes;
                auto directions = positive_cartesian_directions<dim>();
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto& flux = fluxes[d];
                    flux.direction = xt::view(directions, d);
                    flux.computational_stencil = in_out_stencil<dim>(flux.direction);
                    flux.get_coeffs = [](double h_I, double h_F)
                        {
                            double one_over_h2 = 1/(h_I*h_F);
                            auto Identity = eye<coeff_matrix_t>();
                            std::array<coeff_matrix_t, 2> coeffs;
                            coeffs[0] =  one_over_h2 * Identity;
                            coeffs[1] = -one_over_h2 * Identity;
                            return coeffs;
                        };
                }
                return fluxes;
            }

            bool matrix_is_spd() const override
            {
                if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                {
                    // The projections/predictions kill the symmetry, so the matrix is spd only if the mesh is uniform.
                    return this->mesh().min_level() == this->mesh().max_level();
                }
                else
                {
                    return false;
                }
            }
        };


        template<DirichletEnforcement dirichlet_enfcmt=Equation, class Field>
        auto make_diffusion_FV(Field& f)
        {
            return DiffusionFV<Field, dirichlet_enfcmt>(f);
        }

    } // end namespace petsc
} // end namespace samurai