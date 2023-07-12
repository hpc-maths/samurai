#pragma once
#include "../flux_based_scheme.hpp"

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
              // scheme config
              std::size_t dim               = Field::dim,
              std::size_t output_field_size = dim,
              std::size_t stencil_size      = 2,
              class cfg                     = FluxBasedAssemblyConfig<output_field_size, stencil_size>,
              class bdry_cfg                = BoundaryConfigFV<stencil_size / 2>>
    class GradientFV : public FluxBasedScheme<GradientFV<Field>, cfg, bdry_cfg, Field>
    {
        using base_class = FluxBasedScheme<GradientFV<Field>, cfg, bdry_cfg, Field>;

      public:

        using coefficients_t = typename base_class::coefficients_t;
        using coeff_matrix_t = typename coefficients_t::coeff_matrix_t;

        explicit GradientFV(Field& u)
            : base_class(u)
        {
            this->set_name("Gradient");
            static_assert(Field::size == 1, "The field put in the gradient operator must be a scalar field.");
        }

        template <std::size_t d>
        static auto half_flux_in_direction(std::array<double, 2>& flux_coeffs, double h_face, double h_cell)
        {
            std::array<coeff_matrix_t, 2> coeffs;
            coeffs[0].fill(0);
            coeffs[1].fill(0);
            double h_factor        = pow(h_face, dim - 1) / pow(h_cell, dim);
            xt::view(coeffs[0], d) = 0.5 * flux_coeffs[0] * h_factor;
            xt::view(coeffs[1], d) = 0.5 * flux_coeffs[1] * h_factor;
            return coeffs;
        }

        // Grad_x(u) = 1/2 * [ Fx(L) + Fx(R) ]
        // Grad_y(u) = 1/2 * [ Fx(B) + Fx(T) ]
        static auto coefficients()
        {
            static_assert(dim <= 3, "GradientFV.grad_coefficients() not implemented for dim > 3.");
            std::array<coefficients_t, dim> coeffs_by_fluxes;
            auto directions = positive_cartesian_directions<dim>();

            static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
                [&](auto integral_constant_d)
                {
                    static constexpr int d = decltype(integral_constant_d)::value;

                    auto& coeffs                   = coeffs_by_fluxes[d];
                    DirectionVector<dim> direction = xt::view(directions, d);
                    coeffs.flux                    = normal_grad_order2<Field>(direction);
                    coeffs.get_left_cell_coeffs    = half_flux_in_direction<d>;
                    coeffs.get_right_cell_coeffs   = half_flux_in_direction<d>;
                });
            return coeffs_by_fluxes;
        }
    };

    template <class Field>
    auto make_gradient_FV(Field& f)
    {
        return GradientFV<Field>(f);
    }

} // end namespace samurai
