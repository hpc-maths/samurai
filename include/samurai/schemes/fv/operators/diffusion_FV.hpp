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
        using base_class::field_size;

      public:

        using cfg_t                     = cfg;
        using field_t                   = Field;
        using Mesh                      = typename Field::mesh_t;
        using scheme_definition_t       = typename base_class::scheme_definition_t;
        using flux_definition_t         = typename scheme_definition_t::flux_definition_t;
        using scheme_stencil_coeffs_t   = typename scheme_definition_t::scheme_stencil_coeffs_t;
        using flux_stencil_coeffs_t     = typename scheme_definition_t::flux_stencil_coeffs_t;
        using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

        explicit DiffusionFV(const flux_definition_t& flux_definition)
            : base_class(flux_definition)
        {
            this->set_name("Diffusion");
            add_contribution_to_scheme_definition();
        }

      private:

        void add_contribution_to_scheme_definition()
        {
            static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
                [&](auto integral_constant_d)
                {
                    static constexpr int d = decltype(integral_constant_d)::value;
                    this->definition()[d].set_contribution(minus_flux);
                });
        }

        /**
         * To compute Lap(u) in a cell T, we compute the average value 1/|T| * Int_T[ Lap(u) ].
         * By the divergence theorem, we have
         *               1/|T| * Int_T[ Lap(u) ] = 1/|T| * sum_F Int_F[ Grad(u).n ].
         * Here, Grad(u).n is the normal flux, which we denote 'flux' and which we get as a parameter.
         * The contribution of one face F is then
         *               1/|T| * Int_F[ flux ].
         * As the flux is considered constant through the whole face, we finally have the contribution
         *             |F|/|T| * flux.
         * Conclusion: the contribution of the face is just the flux received as a parameter, multiplied by |F|/|T|.
         * Here, we add a minus sign because we define Diffusion as -Lap.
         */
        static scheme_stencil_coeffs_t minus_flux(const flux_stencil_coeffs_t& flux)
        {
            return -flux;
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
            // The projections/predictions kill the symmetry, so the matrix is spd only if the mesh is uniform.
            return is_uniform(unknown.mesh());
        }

        bool matrix_is_spd(const field_t& unknown) const override
        {
            return matrix_is_symmetric(unknown);
        }
    };

    template <DirichletEnforcement dirichlet_enfcmt, class Field>
    auto make_diffusion()
    {
        static constexpr std::size_t flux_output_field_size = Field::size;

        auto flux_definition = make_flux_definition<Field, flux_output_field_size>(get_normal_grad_order1_coeffs<Field>);
        return DiffusionFV<Field, dirichlet_enfcmt>(flux_definition);
        // return make_divergence_FV(flux_definition);
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

    template <class Field, std::size_t flux_output_field_size, std::size_t stencil_size, DirichletEnforcement dirichlet_enfcmt = Equation>
    auto make_diffusion(const FluxDefinition<FluxType::LinearHomogeneous, Field, flux_output_field_size, stencil_size>& flux_definition)
    {
        return DiffusionFV<Field, dirichlet_enfcmt, stencil_size>(flux_definition);
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
