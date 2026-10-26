#pragma once
#include "../FV_scheme.hpp"

namespace samurai
{
    namespace detail
    {
        /**
         * Boundary equation of the -Laplacian, for every boundary direction:
         *                    [ -1/h2  ghost_sign/h2 ] = rhs_factor(h) * boundary_value
         * on the stencil [cell, interior cell, ghost], the interior cell having a zero coefficient.
         */
        template <class DirectionalBdryConfigs, class RhsFactor>
        void set_diffusion_bc_config(DirectionalBdryConfigs& configs, double ghost_sign, RhsFactor rhs_factor)
        {
            using bdry_equation_config_t = typename DirectionalBdryConfigs::value_type::bdry_equation_config_t;
            using stencil_coeffs_t       = typename bdry_equation_config_t::stencil_coeffs_t;
            using coeffs_t               = typename bdry_equation_config_t::equation_coeffs_t::coeffs_t;

            static constexpr std::size_t cell          = 0;
            static constexpr std::size_t interior_cell = 1;
            static constexpr std::size_t ghost         = 2;

            for (auto& config : configs)
            {
                config.equations[0].ghost_index        = ghost;
                config.equations[0].get_stencil_coeffs = [ghost_sign](double h)
                {
                    stencil_coeffs_t coeffs;
                    auto Identity         = eye<coeffs_t>();
                    coeffs[cell]          = -1 / (h * h) * Identity;
                    coeffs[ghost]         = ghost_sign / (h * h) * Identity;
                    coeffs[interior_cell] = zeros<coeffs_t>();
                    return coeffs;
                };
                config.equations[0].get_rhs_coeffs = [rhs_factor](double h)
                {
                    auto Identity   = eye<coeffs_t>();
                    coeffs_t coeffs = rhs_factor(h) * Identity;
                    return coeffs;
                };
            }
        }

        /**
         * Dirichlet boundary equation of the -Laplacian, shared by the flux-based and the cell-based diffusion schemes.
         */
        template <class DirectionalBdryConfigs>
        void set_diffusion_dirichlet_config(DirectionalBdryConfigs& configs)
        {
            // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is
            //                        [  1/2    1/2 ] = dirichlet_value
            // which is equivalent to
            //                        [ -1/h2  -1/h2] = -2/h2 * dirichlet_value
            set_diffusion_bc_config(configs,
                                    -1.,
                                    [](double h)
                                    {
                                        return -2 / (h * h);
                                    });
        }

        /**
         * Neumann boundary equation of the -Laplacian, shared by the flux-based and the cell-based diffusion schemes.
         */
        template <class DirectionalBdryConfigs>
        void set_diffusion_neumann_config(DirectionalBdryConfigs& configs)
        {
            // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is
            //                    [ 1/h  -1/h ] = neumann_value
            // However, to have symmetry, we want to have 1/h2 as the off-diagonal coefficient, so
            //                    [1/h2  -1/h2] = (1/h) * neumann_value
            set_diffusion_bc_config(configs,
                                    1.,
                                    [](double h)
                                    {
                                        return 1 / h;
                                    });
        }
    } // namespace detail
} // namespace samurai
