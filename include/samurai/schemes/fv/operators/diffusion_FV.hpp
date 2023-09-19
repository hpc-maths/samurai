#pragma once
#include "../flux_based_scheme__lin_hom.hpp"
#include "divergence_FV.hpp"

namespace samurai
{
    /**
     * @class DiffusionFV:
     * implements the operator -Laplacian.
     */
    template <
        // template parameters
        class cfg,
        DirichletEnforcement dirichlet_enfcmt = Equation,
        // boundary config
        class bdry_cfg = BoundaryConfigFV<cfg::stencil_size / 2, dirichlet_enfcmt>>
    class DiffusionFV : public FluxBasedScheme<cfg, bdry_cfg>
    {
        using base_class = FluxBasedScheme<cfg, bdry_cfg>;
        using base_class::bdry_stencil_size;
        using base_class::dim;

      public:

        using flux_definition_t         = typename base_class::flux_definition_t;
        using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

        explicit DiffusionFV(const flux_definition_t& flux_definition)
            : base_class(flux_definition)
        {
            this->set_name("Diffusion");
            this->is_symmetric(true);
            this->is_spd(true);

            set_dirichlet_config();
            set_neumann_config();
        }

        //---------------------------------//
        //       Boundary conditions       //
        //---------------------------------//

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

    template <class cfg, DirichletEnforcement dirichlet_enfcmt = Equation>
    auto make_diffusion(const FluxDefinition<cfg>& flux_definition)
    {
        return DiffusionFV<cfg, dirichlet_enfcmt>(flux_definition);
    }

    template <DirichletEnforcement dirichlet_enfcmt, class Field>
    auto make_diffusion()
    {
        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2;

        using cfg = FluxBasedSchemeConfig<FluxType::LinearHomogeneous, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> normal_grad;

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                normal_grad[d].flux_function = [](double h)
                {
                    static constexpr std::size_t left  = 0;
                    static constexpr std::size_t right = 1;

                    // Return value: 2 matrices (left, right) of size output_field_size x field_size.
                    // In this case, of size field_size x field_size.
                    FluxStencilCoeffs<cfg> coeffs;
                    if constexpr (field_size == 1)
                    {
                        coeffs[left]  = -1 / h;
                        coeffs[right] = 1 / h;
                    }
                    else
                    {
                        coeffs[left].fill(0);
                        coeffs[right].fill(0);
                        for (std::size_t i = 0; i < field_size; ++i)
                        {
                            coeffs[left](i, i)  = -1 / h;
                            coeffs[right](i, i) = 1 / h;
                        }
                    }
                    // Because we want -Laplacian
                    coeffs[left] *= -1;
                    coeffs[right] *= -1;
                    return coeffs;
                };
            });

        return make_diffusion<cfg, dirichlet_enfcmt>(normal_grad);
    }

    template <class Field>
    auto make_diffusion()
    {
        return make_diffusion<DirichletEnforcement::Equation, Field>();
    }

    template <class Field>
    [[deprecated("Use make_diffusion() instead.")]] auto make_diffusion_FV()
    {
        return make_diffusion<Field>();
    }

    template <DirichletEnforcement dirichlet_enfcmt, class Field>
    auto make_laplacian()
    {
        return -make_diffusion<dirichlet_enfcmt, Field>();
    }

    template <class Field>
    auto make_laplacian()
    {
        return -make_diffusion<Field>();
    }

} // end namespace samurai
