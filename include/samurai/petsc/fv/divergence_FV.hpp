#pragma once
#include "../flux_based_scheme.hpp"

namespace samurai
{
    namespace petsc
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
                  std::size_t dim            = Field::dim,
                  PetscInt output_field_size = 1,
                  PetscInt stencil_size      = 2,
                  class cfg                  = FluxBasedAssemblyConfig<output_field_size, stencil_size>,
                  class bdry_cfg             = BoundaryConfigFV<1>>
        class DivergenceFV : public FluxBasedScheme<cfg, bdry_cfg, Field>
        {
            using base_class = FluxBasedScheme<cfg, bdry_cfg, Field>;

          public:

            using coefficients_t                    = typename base_class::coefficients_t;
            using flux_matrix_t                     = typename coefficients_t::flux_computation_t::flux_matrix_t;
            using coeff_matrix_t                    = typename coefficients_t::coeff_matrix_t;
            static constexpr std::size_t field_size = Field::size;

            explicit DivergenceFV(Field& u)
                : base_class(u, div_coefficients())
            {
                this->set_name("Divergence");
                static_assert(dim == field_size, "The field put into the divergence operator must have a size equal to the space dimension.");
            }

            template <std::size_t d>
            static auto average(std::array<flux_matrix_t, 2>&, double h_face, double h_cell)
            {
                std::array<coeff_matrix_t, 2> coeffs;
                double h_factor = pow(h_face, dim - 1) / pow(h_cell, dim);
                if constexpr (field_size == 1)
                {
                    coeffs[0] = 0.5 * h_factor;
                    coeffs[1] = 0.5 * h_factor;
                }
                else
                {
                    coeffs[0].fill(0);
                    coeffs[1].fill(0);
                    coeffs[0](d) = 0.5 * h_factor;
                    coeffs[1](d) = 0.5 * h_factor;
                }
                return coeffs;
            }

            template <std::size_t d>
            static auto minus_average(std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
            {
                auto coeffs = average<d>(flux_coeffs, h_face, h_cell);
                for (auto& coeff : coeffs)
                {
                    coeff *= -1;
                }
                return coeffs;
            }

            // Div(F) =  (Fx_{L} + Fx_{R}) / 2  +  (Fy_{B} + Fy_{T}) / 2
            static auto div_coefficients()
            {
                static_assert(dim <= 3, "DivergenceFV.div_coefficients() not implemented for dim > 3.");
                std::array<coefficients_t, dim> coeffs_by_fluxes;
                auto directions = positive_cartesian_directions<dim>();
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto& coeffs                   = coeffs_by_fluxes[d];
                    DirectionVector<dim> direction = xt::view(directions, d);
                    coeffs.flux                    = normal_grad_order2<Field>(direction);
                    if (d == 0)
                    {
                        coeffs.get_cell1_coeffs = average<0>;
                        coeffs.get_cell2_coeffs = minus_average<0>;
                    }
                    if constexpr (dim >= 2)
                    {
                        if (d == 1)
                        {
                            coeffs.get_cell1_coeffs = average<1>;
                            coeffs.get_cell2_coeffs = minus_average<1>;
                        }
                    }
                    if constexpr (dim >= 3)
                    {
                        if (d == 2)
                        {
                            coeffs.get_cell1_coeffs = average<2>;
                            coeffs.get_cell2_coeffs = minus_average<2>;
                        }
                    }
                }
                return coeffs_by_fluxes;
            }
        };

        template <class Field>
        auto make_divergence_FV(Field& f)
        {
            return DivergenceFV<Field>(f);
        }

    } // end namespace petsc
} // end namespace samurai
