#pragma once
#include "../flux_based_scheme.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * @class DiffusionFV
         * Assemble the matrix for the problem -Lap(u)=f.
         * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
         */
        template <class Field,
                  DirichletEnforcement dirichlet_enfcmt = Equation,
                  std::size_t dim                       = Field::dim,
                  std::size_t output_field_size         = Field::size,
                  std::size_t stencil_size              = 2,
                  class cfg                             = FluxBasedAssemblyConfig<output_field_size, stencil_size>,
                  class bdry_cfg                        = BoundaryConfigFV<stencil_size / 2, dirichlet_enfcmt>>
        class DiffusionFV : public FluxBasedScheme<cfg, bdry_cfg, Field>
        {
            using base_class = FluxBasedScheme<cfg, bdry_cfg, Field>;
            using base_class::bdry_stencil_size;
            using base_class::field_size;

          public:

            using cfg_t                     = cfg;
            using field_t                   = Field;
            using Mesh                      = typename Field::mesh_t;
            using coefficients_t            = typename base_class::coefficients_t;
            using flux_matrix_t             = typename coefficients_t::flux_computation_t::flux_matrix_t;
            using coeff_matrix_t            = typename coefficients_t::coeff_matrix_t;
            using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

            explicit DiffusionFV(Field& unknown)
                : base_class(unknown, diffusion_coefficients())
            {
                this->set_name("Diffusion");
            }

            template <std::size_t d>
            static auto get_laplacian_coeffs_cell1(std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
            {
                double h_factor = pow(h_face, dim - 1) / pow(h_cell, dim);
                std::array<coeff_matrix_t, 2> coeffs;
                if constexpr (field_size == 1)
                {
                    coeffs[0] = flux_coeffs[0] * h_factor;
                    coeffs[1] = flux_coeffs[1] * h_factor;
                }
                else
                {
                    coeffs[0].fill(0);
                    coeffs[1].fill(0);
                    for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                    {
                        coeffs[0](field_j, field_j) = flux_coeffs[0](field_j) * h_factor;
                        coeffs[1](field_j, field_j) = flux_coeffs[1](field_j) * h_factor;
                    }
                }
                return coeffs;
            }

            static auto diffusion_coefficients()
            {
                std::array<coefficients_t, dim> coeffs_by_fluxes;
                auto directions = positive_cartesian_directions<dim>();

                static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
                    [&](auto integral_constant_d)
                    {
                        static constexpr int d = decltype(integral_constant_d)::value;

                        auto& coeffs                   = coeffs_by_fluxes[d];
                        DirectionVector<dim> direction = xt::view(directions, d);
                        coeffs.flux                    = normal_grad_order2<Field>(direction);
                        coeffs.get_cell1_coeffs        = [](std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
                        {
                            auto cell_coeffs = get_laplacian_coeffs_cell1<d>(flux_coeffs, h_face, h_cell);
                            for (auto& coeff : cell_coeffs)
                            {
                                coeff *= -1;
                            }
                            return cell_coeffs;
                        };
                        coeffs.get_cell2_coeffs = get_laplacian_coeffs_cell1<d>;
                    });
                return coeffs_by_fluxes;
            }

          protected:

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

          public:

            bool matrix_is_symmetric() const override
            {
                // The projections/predictions kill the symmetry, so the matrix is spd only if the mesh is uniform.
                return is_uniform(this->mesh());
            }

            bool matrix_is_spd() const override
            {
                return matrix_is_symmetric();
            }
        };

        template <DirichletEnforcement dirichlet_enfcmt = Equation, class Field>
        auto make_diffusion_FV(Field& f)
        {
            return DiffusionFV<Field, dirichlet_enfcmt>(f);
        }

    } // end namespace petsc
} // end namespace samurai
