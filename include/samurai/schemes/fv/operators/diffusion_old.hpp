#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

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
              class cfg                       = StarStencilSchemeConfig<SchemeType::LinearHomogeneous, neighbourhood_width, Field, Field>,
              class bdry_cfg                  = BoundaryConfigFV<neighbourhood_width, dirichlet_enfcmt>>
    class DiffusionFV_old : public CellBasedScheme<cfg, bdry_cfg>
    {
        using base_class = CellBasedScheme<cfg, bdry_cfg>;
        using base_class::bdry_stencil_size;
        using base_class::n_comp;

      public:

        using field_value_type          = typename Field::value_type;
        using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

        explicit DiffusionFV_old()
        {
            this->set_name("Diffusion");

            this->stencil()           = star_stencil<dim>();
            this->coefficients_func() = [](double h)
            {
                auto Identity = eye<field_value_type, n_comp, n_comp, Field::is_scalar>();
                StencilCoeffs<cfg> coeffs;
                for (unsigned int i = 0; i < cfg::stencil_size; ++i)
                {
                    coeffs[i] = -Identity;
                }
                coeffs[cfg::center_index] = (cfg::stencil_size - 1) * Identity;
                coeffs /= (h * h);
                return coeffs;
            };

            this->is_symmetric(true);
            this->is_spd(true);
            set_dirichlet_config();
            set_neumann_config();
        }

        void set_dirichlet_config()
        {
            for (std::size_t d = 0; d < 2 * dim; ++d)
            {
                auto& config = this->dirichlet_config()[d];

                using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;

                static constexpr std::size_t cell          = 0;
                static constexpr std::size_t interior_cell = 1;
                static constexpr std::size_t ghost         = 2;

                // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is
                //                        [  1/2    1/2 ] = dirichlet_value
                // which is equivalent to
                //                        [ -1/h2  -1/h2] = -2/h2 * dirichlet_value
                config.equations[0].ghost_index        = ghost;
                config.equations[0].get_stencil_coeffs = [](double h)
                {
                    std::array<coeffs_t, bdry_stencil_size> coeffs;
                    auto Identity         = eye<coeffs_t>();
                    coeffs[cell]          = -1 / (h * h) * Identity;
                    coeffs[ghost]         = -1 / (h * h) * Identity;
                    coeffs[interior_cell] = zeros<coeffs_t>();
                    return coeffs;
                };
                config.equations[0].get_rhs_coeffs = [](double h)
                {
                    coeffs_t coeffs;
                    auto Identity = eye<coeffs_t>();
                    coeffs        = -2 / (h * h) * Identity;
                    return coeffs;
                };
            }
        }

        void set_neumann_config()
        {
            for (std::size_t d = 0; d < 2 * dim; ++d)
            {
                auto& config = this->neumann_config()[d];

                using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;

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
            }
        }
    };

    template <class Field, DirichletEnforcement dirichlet_enfcmt = Equation>
    auto make_diffusion_old()
    {
        return DiffusionFV_old<Field, dirichlet_enfcmt>();
    }

} // end namespace samurai
