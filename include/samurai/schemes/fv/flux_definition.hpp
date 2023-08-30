#pragma once

namespace samurai
{
    /**
     * Defines how to compute a normal flux
     */
    template <class Field, std::size_t stencil_size = 2, bool is_linear = false, bool is_heterogeneous = true>
    struct NormalFluxDefinition
    {
    };

    /**
     * Defines how to compute a LINEAR and HOMOGENEOUS normal flux
     */
    template <class Field, std::size_t stencil_size>
    struct NormalFluxDefinition<Field, stencil_size, true, false>
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
         *            auto flux_function(double h)
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
        flux_func flux_function;

        ~NormalFluxDefinition()
        {
            flux_function = nullptr;
        }
    };

    /**
     * Defines how to compute a NON-LINEAR normal flux
     */
    template <class Field, std::size_t stencil_size>
    struct NormalFluxDefinition<Field, stencil_size, false, true>
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type;
        using cell_t                            = typename Field::cell_t;
        using stencil_cells_t                   = std::array<cell_t, stencil_size>;
        using flux_value_t                      = typename detail::LocalMatrix<field_value_type, field_size, 1>::Type;
        using flux_func                         = std::function<flux_value_t(Field&, stencil_cells_t&)>;

        DirectionVector<dim> direction;
        Stencil<stencil_size, dim> stencil;
        flux_func flux_function;

        ~NormalFluxDefinition()
        {
            flux_function = nullptr;
        }
    };

    /**
     * @class FluxDefinition
     * Stores one object of @class NormalFluxDefinition for each positive Cartesian direction.
     */
    template <class Field, std::size_t stencil_size, bool is_linear, bool is_heterogeneous>
    class FluxDefinition
    {
      public:

        static constexpr std::size_t dim  = Field::dim;
        using flux_computation_t          = NormalFluxDefinition<Field, stencil_size, is_linear, is_heterogeneous>;
        using flux_computation_stencil2_t = NormalFluxDefinition<Field, 2, is_linear, is_heterogeneous>;

      private:

        std::array<flux_computation_t, dim> m_normal_fluxes;

      public:

        FluxDefinition()
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
                m_normal_fluxes[d].flux_function = nullptr; // to be set by the user
            }
        }

        /**
         * This constructor sets the same flux function for all directions
         */
        FluxDefinition(typename flux_computation_stencil2_t::flux_func flux_implem)
        {
            static_assert(stencil_size == 2, "stencil_size = 2 required to use this constructor.");

            auto directions = positive_cartesian_directions<dim>();
            for (std::size_t d = 0; d < dim; ++d)
            {
                DirectionVector<dim> direction   = xt::view(directions, d);
                m_normal_fluxes[d].direction     = direction;
                m_normal_fluxes[d].stencil       = in_out_stencil<dim>(direction);
                m_normal_fluxes[d].flux_function = flux_implem;
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

    /**
     * Defines a LINEAR and HOMOGENEOUS flux
     */
    template <class Field, std::size_t stencil_size = 2>
    auto make_flux_definition(typename NormalFluxDefinition<Field, stencil_size, true, false>::flux_func flux_impl)
    {
        return FluxDefinition<Field, stencil_size, true, false>(flux_impl);
    }

    /**
     * Defines a LINEAR and HETEROGENEOUS flux
     */
    template <class Field, std::size_t stencil_size = 2>
    auto make_flux_definition(typename NormalFluxDefinition<Field, stencil_size, true, true>::flux_func flux_impl)
    {
        return FluxDefinition<Field, stencil_size, true, true>(flux_impl);
    }

    /**
     * Defines a NON-LINEAR flux
     */
    template <class Field, std::size_t stencil_size = 2>
    auto make_flux_definition(typename NormalFluxDefinition<Field, stencil_size, false, true>::flux_func flux_impl)
    {
        return FluxDefinition<Field, stencil_size, false, true>(flux_impl);
    }

} // end namespace samurai
