#pragma once
#include "../flux_based_scheme__lin_hom.hpp"
#include "divergence_FV.hpp"

namespace samurai
{
    /**
     * @class DiffusionFV
     * Assemble the matrix for the problem -Lap(u)=f.
     * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
     */
    template <
        // template parameters
        class Field,
        DirichletEnforcement dirichlet_enfcmt = Equation,
        std::size_t stencil_size              = 2,
        // scheme config
        std::size_t dim               = Field::dim,
        std::size_t output_field_size = Field::size,
        class cfg                     = FluxBasedSchemeConfig<FluxType::LinearHomogeneous, output_field_size, stencil_size>,
        class bdry_cfg                = BoundaryConfigFV<stencil_size / 2, dirichlet_enfcmt>>
    class DiffusionFV : public FluxBasedScheme<DiffusionFV<Field, dirichlet_enfcmt, stencil_size>, cfg, bdry_cfg, Field>
    {
        using base_class = FluxBasedScheme<DiffusionFV<Field, dirichlet_enfcmt, stencil_size>, cfg, bdry_cfg, Field>;
        using base_class::bdry_stencil_size;

      public:

        using field_t                   = Field;
        using scheme_definition_t       = typename base_class::scheme_definition_t;
        using flux_definition_t         = typename scheme_definition_t::flux_definition_t;
        using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

        explicit DiffusionFV(const flux_definition_t& flux_definition)
            : base_class(flux_definition)
        {
            this->set_name("Diffusion");
        }

        //---------------------------------//
        //       Boundary conditions       //
        //---------------------------------//

      public:

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
            // The projections/predictions kill the symmetry, so the matrix is spd only if the mesh is uniform.
            return is_uniform(unknown.mesh());
        }

        bool matrix_is_spd(const field_t& unknown) const override
        {
            return matrix_is_symmetric(unknown);
        }
    };

    template <class Field, std::size_t flux_output_field_size, std::size_t stencil_size = 2, DirichletEnforcement dirichlet_enfcmt = Equation>
    auto make_diffusion(const FluxDefinition<FluxType::LinearHomogeneous, Field, flux_output_field_size, stencil_size>& flux_definition)
    {
        return DiffusionFV<Field, dirichlet_enfcmt, stencil_size>(flux_definition);
    }

    template <DirichletEnforcement dirichlet_enfcmt, class Field>
    auto make_diffusion()
    {
        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;

        using flux_computation_t = NormalFluxDefinition<FluxType::LinearHomogeneous, Field, output_field_size>;

        // 2 matrices (left, right) of size output_field_size x field_size.
        // In the case of the laplacian, of size field_size x field_size.
        using flux_stencil_coeffs_t        = typename flux_computation_t::flux_stencil_coeffs_t;
        static constexpr std::size_t left  = 0;
        static constexpr std::size_t right = 1;

        auto normal_grad = samurai::make_flux_definition<FluxType::LinearHomogeneous, Field, output_field_size>();

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                normal_grad[d].flux_function = [](double h)
                {
                    flux_stencil_coeffs_t coeffs;
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

        return make_diffusion<Field, output_field_size, 2, dirichlet_enfcmt>(normal_grad);
        //  return make_divergence(flux_definition);
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
