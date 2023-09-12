#pragma once
#include "../flux_based_scheme__lin_hom.hpp"

namespace samurai
{
    /**
     * If u is a scalar field in dimension 2, then
     *      Grad(u) = [d(u)/dx]
     *                [d(u)/dy].
     * On each cell, we adopt a cell-centered approximation:
     *         d(u)/dx = 1/2 [(u^R - u)/h + (u - u^L)/h]   where (L, R) = (Left, Right)
     *         d(u)/dy = 1/2 [(u^T - u)/h + (u - u^B)/h]   where (B, T) = (Bottom, Top).
     * We denote by Fx^f = d(u)/dx * n_f the outer normal flux through the face f, i.e.
     *         Fx^R = (u^R - u)/h,      Fx^L = (u^L - u)/h,
     *         Fy^T = (u^T - u)/h,      Fy^B = (u^B - u)/h.
     * The approximations become
     *         d(u)/dx = 1/2 (Fx^R - Fx^L)        (S)
     *         d(u)/dy = 1/2 (Fy^T - Fy^B).
     *
     * Implementation:
     *
     * 1. The computation of the normal fluxes between cell 1 and cell 2 (in the direction dir=x or y) is given by
     *         Fx = (u^2 - u^1)/h = -1/h * u^1 + 1/h * u^2    if dir=x
     *         Fy = (u^2 - u^1)/h = -1/h * u^1 + 1/h * u^2    if dir=y
     *
     *    So   F = [ -1/h |  1/h ] whatever the direction
     *             |______|______|
     *              cell 1 cell 2
     *
     * 2. On each couple (cell 1, cell 2), we compute Fx^R(1) or Fx^T(1) (according to the direction) and consider that
     *         Fx^L(1) = -Fx^R(2)
     *         Fy^B(1) = -Fy^T(2).
     *    The gradient scheme (S) becomes
     *         Grad(u)(cell 1) = [1/2 (Fx^R(1) + Fx^L(2))]
     *                           [1/2 (Fy^T(1) + Fy^B(2))], where 2 denotes the neighbour in the appropriate direction.
     *    So the contribution of a flux F (R or T) computed on cell 1 is
     *         for cell 1: [1/2 F] if dir=x,  [    0] if dir=y
     *                     [    0]            [1/2 F]
     *         for cell 2: [1/2 F] if dir=x,  [    0] if dir=y
     *                     [    0]            [1/2 F]
     */
    template <class Field,
              std::size_t stencil_size = 2,
              // scheme config
              std::size_t dim               = Field::dim,
              std::size_t output_field_size = dim,
              class cfg                     = FluxBasedSchemeConfig<FluxType::LinearHomogeneous, output_field_size, stencil_size>,
              class bdry_cfg                = BoundaryConfigFV<stencil_size / 2>>
    class GradientFV : public FluxBasedScheme<GradientFV<Field, stencil_size>, cfg, bdry_cfg, Field>
    {
        using base_class = FluxBasedScheme<GradientFV<Field, stencil_size>, cfg, bdry_cfg, Field>;

      public:

        using scheme_definition_t     = typename base_class::scheme_definition_t;
        using flux_definition_t       = typename scheme_definition_t::flux_definition_t;
        using scheme_stencil_coeffs_t = typename scheme_definition_t::scheme_stencil_coeffs_t;
        using flux_stencil_coeffs_t   = typename scheme_definition_t::flux_stencil_coeffs_t;

        explicit GradientFV(const flux_definition_t& flux_definition)
            : base_class(flux_definition)
        {
            this->set_name("Gradient");
            static_assert(Field::size == 1, "The field put in the gradient operator must be a scalar field.");
            add_contribution_to_scheme_definition();
        }

      private:

        void add_contribution_to_scheme_definition()
        {
            static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
                [&](auto integral_constant_d)
                {
                    static constexpr int d = decltype(integral_constant_d)::value;
                    this->definition()[d].set_contribution(add_flux_to_row<d>);
                });
        }

        template <std::size_t d>
        static scheme_stencil_coeffs_t add_flux_to_row(flux_stencil_coeffs_t& flux)
        {
            scheme_stencil_coeffs_t coeffs;
            for (std::size_t i = 0; i < stencil_size; ++i)
            {
                if constexpr (dim == 1)
                {
                    coeffs[i] = flux[i];
                }
                else
                {
                    coeffs[i].fill(0);
                    xt::row(coeffs[i], d) = flux[i];
                }
            }
            return coeffs;
        }
    };

    template <class Field>
    auto make_gradient()
    {
        static constexpr std::size_t flux_output_field_size = Field::size;

        auto flux_definition = make_flux_definition<Field, flux_output_field_size>(get_average_coeffs<Field>);
        return make_gradient(flux_definition);
    }

    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    auto make_gradient(const FluxDefinition<FluxType::LinearHomogeneous, Field, output_field_size, stencil_size>& flux_definition)
    {
        return GradientFV<Field, stencil_size>(flux_definition);
    }

    template <class Field>
    [[deprecated("Use make_gradient() instead.")]] auto make_gradient_FV()
    {
        return make_gradient<Field>();
    }

} // end namespace samurai
