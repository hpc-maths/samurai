#pragma once
#include "../flux_based_scheme.hpp"

namespace samurai
{
    /**
     * If u is a field of size 2, e.g. the velocity --> u = (u_x, u_y), then
     *         Div(u) = d(u_x)/dx + d(u_y)/dy.
     * On each cell, we adopt a cell-centered approximation:
     *         d(u_x)/dx = 1/2 [(u_x^R - u_x)/h + (u_x - u_x^L)/h]   where (L, R) = (Left, Right)
     *         d(u_y)/dy = 1/2 [(u_y^T - u_y)/h + (u_y - u_x^B)/h]   where (B, T) = (Bottom, Top).
     * We denote by Fx^f = d(u_x)/dx * n_f the outer normal flux through the face f, i.e.
     *         Fx^R = (u_x^R - u_x)/h,      Fx^L = (u_x^L - u_x)/h,
     *         Fy^T = (u_y^T - u_y)/h,      Fy^B = (u_y^B - u_y)/h.
     * The approximations become
     *         d(u_x)/dx = 1/2 (Fx^R - Fx^L)
     *         d(u_y)/dy = 1/2 (Fy^T - Fy^B).
     * and finally,
     *         Div(u) = 1/2 (Fx^R - Fx^L) + 1/2 (Fy^T - Fy^B)      (S)
     *
     * Implementation:
     *
     * 1. The computation of the normal fluxes between cell 1 and cell 2 (in the direction d=x or y) is given by
     *         Fx = (u_x^2 - u_x^1)/h = -1/h * u_x^1 + 1/h * u_x^2 +  0 * u_y^1 +   0 * u_y^2    if d=x
     *         Fy = (u_y^2 - u_y^1)/h =    0 * u_x^1 +   0 * u_x^2 -1/h * u_y^1 + 1/h * u_x^2    if d=y
     *
     *    So   F = [-1/h   0 | 1/h   0  ] if d=x
     *         F = [  0  -1/h|  0   1/h ] if d=y
     *             |_________|__________|
     *               cell 1     cell 2
     *
     * 2. On each couple (cell 1, cell 2), we compute Fx^R(1) or Fx^T(1) (according to the direction) and consider that
     *         Fx^L(2) = -Fx^R(1)
     *         Fy^B(2) = -Fy^T(1).
     *    The divergence scheme (S) becomes
     *         Div(u)(cell 1) = 1/2 (Fx^R(1) + Fx^L(2)) + 1/2 (Fy^T(1) + Fy^B(2)), where 2 denotes the neighbour in the appropriate
     * direction. So the contribution of a flux F (R or T) computed on cell 1 is for cell 1: 1/2 F for cell 2: 1/2 F
     */
    template <class Field,
              // scheme config
              std::size_t dim               = Field::dim,
              std::size_t output_field_size = 1,
              std::size_t stencil_size      = 2,
              class cfg                     = FluxBasedAssemblyConfig<output_field_size, stencil_size>,
              class bdry_cfg                = BoundaryConfigFV<stencil_size / 2>>
    class DivergenceFV : public FluxBasedScheme<DivergenceFV<Field>, cfg, bdry_cfg, Field>
    {
        using base_class = FluxBasedScheme<DivergenceFV<Field>, cfg, bdry_cfg, Field>;

      public:

        using coefficients_t                    = typename base_class::coefficients_t;
        using cell_coeffs_t                     = typename coefficients_t::cell_coeffs_t;
        using flux_coeffs_t                     = typename coefficients_t::flux_coeffs_t;
        static constexpr std::size_t field_size = Field::size;

        explicit DivergenceFV(Field& u)
            : base_class(u)
        {
            this->set_name("Divergence");
            static_assert(dim == field_size, "The field put into the divergence operator must have a size equal to the space dimension.");
        }

        template <std::size_t d>
        static cell_coeffs_t add_flux_to_col(flux_coeffs_t& flux, double h_face, double h_cell)
        {
            double face_measure = pow(h_face, dim - 1);
            double cell_measure = pow(h_cell, dim);
            double h_factor     = face_measure / cell_measure;

            cell_coeffs_t coeffs;
            for (std::size_t i = 0; i < stencil_size; ++i)
            {
                if constexpr (field_size == 1)
                {
                    coeffs[i] = flux[i] * h_factor;
                }
                else
                {
                    coeffs[i].fill(0);
                    for (std::size_t d2 = 0; d2 < dim; ++d2)
                    {
                        xt::col(coeffs[i], d) += flux[i](d, d2) * h_factor;
                    }
                }
            }
            return coeffs;
        }

        static auto coefficients()
        {
            std::array<coefficients_t, dim> coeffs_by_fluxes;
            auto directions = positive_cartesian_directions<dim>();

            static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
                [&](auto integral_constant_d)
                {
                    static constexpr int d = decltype(integral_constant_d)::value;

                    auto& coeffs                         = coeffs_by_fluxes[d];
                    DirectionVector<dim> direction       = xt::view(directions, d);
                    coeffs.flux                          = average_quantity<Field>(direction);
                    coeffs.get_coeffs                    = add_flux_to_col<d>;
                    coeffs.get_coeffs_opposite_direction = add_flux_to_col<d>;
                });
            return coeffs_by_fluxes;
        }
    };

    template <class Field>
    auto make_divergence_FV(Field& f)
    {
        return DivergenceFV<Field>(f);
    }

} // end namespace samurai
