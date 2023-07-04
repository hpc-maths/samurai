#pragma once
#include "../cell_based_scheme.hpp"

namespace samurai
{
    /**
     * @class DiffusionFV
     * Assemble the matrix for the problem -Lap(u)=f.
     * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
     */
    template <class Field,
              DirichletEnforcement dirichlet_enfcmt = Equation,
              // scheme config
              std::size_t dim                 = Field::dim,
              std::size_t neighbourhood_width = 1,
              class cfg                       = StarStencilFV<dim, Field::size, neighbourhood_width>,
              class bdry_cfg                  = BoundaryConfigFV<neighbourhood_width, dirichlet_enfcmt>>
    class DiffusionFV_old : public CellBasedScheme<DiffusionFV_old<Field, dirichlet_enfcmt>, cfg, bdry_cfg, Field>
    {
        using base_class = CellBasedScheme<DiffusionFV_old<Field, dirichlet_enfcmt>, cfg, bdry_cfg, Field>;
        using base_class::bdry_stencil_size;

      public:

        using field_t                   = Field;
        using Mesh                      = typename Field::mesh_t;
        using local_matrix_t            = typename base_class::local_matrix_t;
        using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

        explicit DiffusionFV_old(Field& unknown)
            : base_class(unknown)
        {
        }

        static constexpr auto stencil()
        {
            return star_stencil<dim>();
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

        directional_bdry_config_t dirichlet_config(const DirectionVector<dim>& direction) const override
        {
            using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;
            directional_bdry_config_t config;

            config.directional_stencil = this->get_directional_stencil(direction);

            static constexpr std::size_t cell          = 0;
            static constexpr std::size_t interior_cell = 1;
            static constexpr std::size_t ghost         = 2;

            // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is
            //                        [  1/2    1/2 ] = dirichlet_value
            // which is equivalent to
            //                        [ -1/h2  -1/h2] = -2/h2 * dirichlet_value
            config.equations[0].ghost_index        = ghost;
            config.equations[0].get_stencil_coeffs = [&](double h)
            {
                std::array<coeffs_t, bdry_stencil_size> coeffs;
                auto Identity         = eye<coeffs_t>();
                coeffs[cell]          = -1 / (h * h) * Identity;
                coeffs[ghost]         = -1 / (h * h) * Identity;
                coeffs[interior_cell] = zeros<coeffs_t>();
                return coeffs;
            };
            config.equations[0].get_rhs_coeffs = [&](double h)
            {
                coeffs_t coeffs;
                auto Identity = eye<coeffs_t>();
                coeffs        = -2 / (h * h) * Identity;
                return coeffs;
            };

            return config;
        }

        directional_bdry_config_t neumann_config(const DirectionVector<dim>& direction) const override
        {
            using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;
            directional_bdry_config_t config;

            config.directional_stencil = this->get_directional_stencil(direction);

            static constexpr std::size_t cell          = 0;
            static constexpr std::size_t interior_cell = 1;
            static constexpr std::size_t ghost         = 2;

            // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is
            //                    [ 1/h  -1/h ] = neumann_value
            // However, to have symmetry, we want to have 1/h2 as the off-diagonal coefficient, so
            //                    [1/h2  -1/h2] = (1/h) * neumann_value
            config.equations[0].ghost_index        = ghost;
            config.equations[0].get_stencil_coeffs = [&](double h)
            {
                std::array<coeffs_t, bdry_stencil_size> coeffs;
                auto Identity         = eye<coeffs_t>();
                coeffs[cell]          = -1 / (h * h) * Identity;
                coeffs[ghost]         = 1 / (h * h) * Identity;
                coeffs[interior_cell] = zeros<coeffs_t>();
                return coeffs;
            };
            config.equations[0].get_rhs_coeffs = [&](double h)
            {
                auto Identity   = eye<coeffs_t>();
                coeffs_t coeffs = (1 / h) * Identity;
                return coeffs;
            };

            return config;
        }

        bool matrix_is_symmetric(const field_t& unknown) const override
        {
            // if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
            //  {
            //  The projections/predictions kill the symmetry, so the matrix is spd only if the mesh is uniform.
            return is_uniform(unknown.mesh());
            // }
            // else
            // {
            //     return false;
            // }
        }

        bool matrix_is_spd(const field_t& unknown) const override
        {
            return matrix_is_symmetric(unknown);
        }
    };

    template <DirichletEnforcement dirichlet_enfcmt = Equation, class Field>
    auto make_diffusion_FV_old(Field& f)
    {
        return DiffusionFV_old<Field, dirichlet_enfcmt>(f);
    }

} // end namespace samurai
