#pragma once
#include "FV_scheme.hpp"

namespace samurai
{
    /**
     * Defines how to compute a normal flux
     */
    template <class Field, std::size_t stencil_size>
    struct LinearNormalFluxDefinition
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type;                               // double
        using flux_matrix_t = typename detail::LocalMatrix<field_value_type, field_size, field_size>::Type; // 'double' if field_size = 1,
                                                                                                            // 'xtensor' representing a
                                                                                                            // matrix otherwise
        using flux_coeffs_t = xt::xtensor_fixed<flux_matrix_t, xt::xshape<stencil_size>>;

        /**
         * Direction of the flux.
         * In 2D, e.g., {1,0} for the flux on the right.
         */
        DirectionVector<dim> direction;

        /**
         * Stencil for the flux computation of the flux in the direction defined above.
         * E.g., if direction = {1,0}, the standard stencil is {{0,0}, {1,0}}.
         * Here, {0,0} captures the current cell and {1,0} its right neighbour.
         * The flux will be computed from {0,0} to {1,0}:
         *
         *                |-------|-------|
         *                | {0,0} | {1,0} |
         *                |-------|-------|
         *                     ------->
         *                    normal flux
         *
         * An enlarged stencil would be {{-1,0}, {0,0}, {1,0}, {1,0}}, i.e. two cells on each side of the interface.
         *
         *       |-------|-------|-------|-------|
         *       |{-1,0} | {0,0} | {1,0} | {2,0} |
         *       |-------|-------|-------|-------|
         *                    ------->
         *                   normal flux
         *
         */
        Stencil<stencil_size, dim> stencil;

        /**
         * Function returning the coefficients for the computation of the flux w.r.t. the defined stencil, in function of the meshsize h.
         * Note that in this definition, the flux must be linear with respect to the cell values.
         * For instance, considering a scalar field u, we configure the flux Grad(u).n through the function
         *
         *            // Grad(u).n = (u_1 - u_0)/h
         *            auto get_flux_coeffs(double h)
         *            {
         *                std::array<double, 2> coeffs;
         *                coeffs[0] = -1/h; // current cell    (because, stencil[0] = {0,0})
         *                coeffs[1] =  1/h; // right neighbour (because, stencil[1] = {1,0})
         *                return coeffs;
         *            }
         * If u is now a vectorial field of size S, then coeffs[0] and coeffs[1] become matrices of size SxS.
         * If the field components are independent from each other, then
         *                coeffs[0] = diag(-1/h),
         *                coeffs[1] = diag( 1/h).
         */
        std::function<flux_coeffs_t(double)> get_flux_coeffs;
    };

    template <class Field, class Vector>
    auto normal_grad_order1(Vector& direction)
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;
        using flux_computation_t                = LinearNormalFluxDefinition<Field, 2>;
        using flux_coeffs_t                     = typename flux_computation_t::flux_coeffs_t;

        flux_computation_t normal_grad;
        normal_grad.direction       = direction;
        normal_grad.stencil         = in_out_stencil<dim>(direction);
        normal_grad.get_flux_coeffs = [](double h)
        {
            flux_coeffs_t coeffs;
            if constexpr (field_size == 1)
            {
                coeffs[0] = -1 / h;
                coeffs[1] = 1 / h;
            }
            else
            {
                coeffs[0].fill(0);
                coeffs[1].fill(0);
                for (std::size_t i = 0; i < field_size; ++i)
                {
                    coeffs[0](i, i) = -1 / h;
                    coeffs[1](i, i) = 1 / h;
                }
            }
            return coeffs;
        };
        return normal_grad;
    }

    template <class Field>
    auto normal_grad_order1()
    {
        static constexpr std::size_t dim = Field::dim;
        using flux_computation_t         = LinearNormalFluxDefinition<Field, 2>;

        auto directions = positive_cartesian_directions<dim>();
        std::array<flux_computation_t, dim> normal_fluxes;
        for (std::size_t d = 0; d < dim; ++d)
        {
            normal_fluxes[d] = normal_grad_order1<Field>(xt::view(directions, d));
        }
        return normal_fluxes;
    }

    template <class Field, class Vector>
    auto average_quantity(Vector& direction)
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;
        using flux_computation_t                = LinearNormalFluxDefinition<Field, 2>;
        using flux_coeffs_t                     = typename flux_computation_t::flux_coeffs_t;

        flux_computation_t flux;
        flux.direction       = direction;
        flux.stencil         = in_out_stencil<dim>(direction);
        flux.get_flux_coeffs = [](double)
        {
            flux_coeffs_t coeffs;
            if constexpr (field_size == 1)
            {
                coeffs[0] = 0.5;
                coeffs[1] = 0.5;
            }
            else
            {
                coeffs[0].fill(0);
                coeffs[1].fill(0);
                for (std::size_t i = 0; i < field_size; ++i)
                {
                    coeffs[0](i, i) = 0.5;
                    coeffs[1](i, i) = 0.5;
                }
            }
            return coeffs;
        };
        return flux;
    }

} // end namespace samurai
