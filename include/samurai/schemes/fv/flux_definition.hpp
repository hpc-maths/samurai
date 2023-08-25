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
        using field_value_type                  = typename Field::value_type;                                     // double
        using flux_coeff_matrix_t = typename detail::LocalMatrix<field_value_type, field_size, field_size>::Type; // 'double' if field_size
                                                                                                                  // = 1, 'xtensor'
                                                                                                                  // representing a matrix
                                                                                                                  // otherwise
        using flux_stencil_coeffs_t = xt::xtensor_fixed<flux_coeff_matrix_t, xt::xshape<stencil_size>>;
        using flux_func             = std::function<flux_stencil_coeffs_t(double)>;

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
        flux_func get_flux_coeffs;
    };

    /**
     * @class LinearFluxDefinition
     * Stores one object of @class LinearNormalFluxDefinition for each positive Cartesian direction.
     */
    template <class Field, std::size_t stencil_size>
    class LinearFluxDefinition
    {
      public:

        static constexpr std::size_t dim = Field::dim;
        using flux_computation_t         = LinearNormalFluxDefinition<Field, stencil_size>;

      private:

        std::array<flux_computation_t, dim> m_normal_fluxes;

      public:

        LinearFluxDefinition()
        {
            auto directions = positive_cartesian_directions<dim>();
            for (std::size_t d = 0; d < dim; ++d)
            {
                DirectionVector<dim> direction = xt::view(directions, d);
                m_normal_fluxes[d].direction   = direction;
                if constexpr (stencil_size == 2)
                {
                    m_normal_fluxes[d].stencil = in_out_stencil<dim>(direction); // TODO: stencil for any stencil size
                }
                m_normal_fluxes[d].get_flux_coeffs = nullptr; // to be set by the user
            }
        }

        /**
         * This constructor sets the same flux function for all directions
         */
        LinearFluxDefinition(typename LinearNormalFluxDefinition<Field, 2>::flux_func flux_implem)
        {
            static_assert(stencil_size == 2, "stencil_size = 2 required to use this constructor.");

            auto directions = positive_cartesian_directions<dim>();
            for (std::size_t d = 0; d < dim; ++d)
            {
                DirectionVector<dim> direction     = xt::view(directions, d);
                m_normal_fluxes[d].direction       = direction;
                m_normal_fluxes[d].stencil         = in_out_stencil<dim>(direction);
                m_normal_fluxes[d].get_flux_coeffs = flux_implem;
            }
        }

        flux_computation_t& operator[](std::size_t d)
        {
            assert(d < dim);
            return m_normal_fluxes[d];
        }

        const flux_computation_t& operator[](std::size_t d) const
        {
            assert(d < dim);
            return m_normal_fluxes[d];
        }
    };

    //-----------------------------------------//
    //          Useful flux functions          //
    //-----------------------------------------//

    /**
     *   |---------|--------|
     *   |         |        |
     *   | cell 0  | cell 1 |
     *   |         |        |
     *   |---------|--------|
     *          ------->
     *        normal flux
     */

    template <class Field>
    auto get_normal_grad_order1_coeffs(double h)
    {
        static constexpr std::size_t field_size = Field::size;
        using flux_computation_t                = LinearNormalFluxDefinition<Field, 2>;
        using flux_stencil_coeffs_t             = typename flux_computation_t::flux_stencil_coeffs_t;

        flux_stencil_coeffs_t coeffs;
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
    }

    // template <class Field, class Vector>
    // auto normal_grad_order1(Vector& direction)
    // {
    //     static constexpr std::size_t dim = Field::dim;
    //     using flux_computation_t         = LinearNormalFluxDefinition<Field, 2>;

    //     flux_computation_t normal_grad;
    //     normal_grad.direction       = direction;
    //     normal_grad.stencil         = in_out_stencil<dim>(direction);
    //     normal_grad.get_flux_coeffs = get_normal_grad_order1_coeffs<Field>;
    //     return normal_grad;
    // }

    template <class Field>
    auto get_average_coeffs(double)
    {
        static constexpr std::size_t field_size = Field::size;
        using flux_computation_t                = LinearNormalFluxDefinition<Field, 2>;
        using flux_stencil_coeffs_t             = typename flux_computation_t::flux_stencil_coeffs_t;

        flux_stencil_coeffs_t coeffs;
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
    }

    // template <class Field, class Vector>
    // auto average_quantity(Vector& direction)
    // {
    //     static constexpr std::size_t dim = Field::dim;
    //     using flux_computation_t         = LinearNormalFluxDefinition<Field, 2>;

    //     flux_computation_t flux;
    //     flux.direction       = direction;
    //     flux.stencil         = in_out_stencil<dim>(direction);
    //     flux.get_flux_coeffs = get_average_coeffs<Field>;
    //     return flux;
    // }

} // end namespace samurai
