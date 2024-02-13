#pragma once
#include "../utils.hpp"
#include <functional>

namespace samurai
{
    template <SchemeType scheme_type_, std::size_t output_field_size_, std::size_t stencil_size_, class InputField_>
    struct FluxConfig
    {
        static constexpr SchemeType scheme_type        = scheme_type_;
        static constexpr std::size_t output_field_size = output_field_size_;
        static constexpr std::size_t stencil_size      = stencil_size_;
        using input_field_t                            = std::decay_t<InputField_>;
        static constexpr std::size_t dim               = input_field_t::dim;
    };

    template <class cfg>
    struct NormalFluxDefinitionBase
    {
        /**
         * Direction of the flux.
         * In 2D, e.g., {1,0} for the flux on the right.
         */
        DirectionVector<cfg::dim> direction;

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
        Stencil<cfg::stencil_size, cfg::dim> stencil;
    };

    /**
     * @class NormalFluxDefinition defines how to compute a normal flux.
     * This struct inherits from @class NormalFluxDefinitionBase and is specialized for all flux types (see below).
     */
    template <class cfg, class enable = void>
    struct NormalFluxDefinition
    {
    };

    /**
     * Specialization of @class NormalFluxDefinition.
     * Defines how to compute a NON-LINEAR normal flux.
     */
    template <class cfg>
    struct NormalFluxDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>> : NormalFluxDefinitionBase<cfg>
    {
        using field_t          = typename cfg::input_field_t;
        using field_value_type = typename field_t::value_type;
        using cell_t           = typename field_t::cell_t;

        using stencil_cells_t = std::array<cell_t, cfg::stencil_size>;

        using flux_value_t      = CollapsVector<field_value_type, cfg::output_field_size>;
        using flux_value_pair_t = xt::xtensor_fixed<flux_value_t, xt::xshape<2>>;
        using flux_func         = std::function<flux_value_pair_t(stencil_cells_t&, const field_t&)>; // non-conservative
        using cons_flux_func    = std::function<flux_value_t(stencil_cells_t&, const field_t&)>;      // conservative

        // using flux_jac_t         = CollapsMatrix<field_value_type, cfg::output_field_size, field_size>;
        // using flux_jacobian_func = std::function<flux_jac_t(stencil_cells_t&, field_t&)>;

        /**
         * Conservative flux function:
         * @returns the flux in the positive direction.
         */
        cons_flux_func cons_flux_function = nullptr;

        /**
         * Non-conservative flux function:
         * @returns the flux in the positive direction and in the negative direction.
         * By default, uses the conservative formula.
         */
        flux_func flux_function = nullptr;

        // flux_jacobian_func flux_jac_function = nullptr;

        /**
         * @returns the non-conservative flux function that calls the conservative one.
         * This function is used to default 'flux_function' if it is not set.
         */
        flux_func flux_function_as_conservative() const
        {
            return [&](auto& cells, const auto& field)
            {
                flux_value_pair_t fluxes;
                fluxes[0] = cons_flux_function(cells, field);
                fluxes[1] = -fluxes[0];
                return fluxes;
            };
        }

        ~NormalFluxDefinition()
        {
            cons_flux_function = nullptr;
            flux_function      = nullptr;
            // flux_jac_function = nullptr;
        }
    };

    /**
     * Specialization of @class NormalFluxDefinition.
     * Defines how to compute a LINEAR and HETEROGENEOUS normal flux.
     */
    template <class cfg>
    struct NormalFluxDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHeterogeneous>> : NormalFluxDefinitionBase<cfg>
    {
        using field_t                           = typename cfg::input_field_t;
        using field_value_type                  = typename field_t::value_type;
        using cell_t                            = typename field_t::cell_t;
        static constexpr std::size_t field_size = field_t::size;

        using stencil_cells_t       = std::array<cell_t, cfg::stencil_size>;
        using flux_coeff_matrix_t   = CollapsMatrix<field_value_type, cfg::output_field_size, field_size>;
        using flux_stencil_coeffs_t = xt::xtensor_fixed<flux_coeff_matrix_t, xt::xshape<cfg::stencil_size>>;
        using cons_flux_func        = std::function<flux_stencil_coeffs_t(stencil_cells_t&)>;

        cons_flux_func cons_flux_function = nullptr;

        ~NormalFluxDefinition()
        {
            cons_flux_function = nullptr;
        }
    };

    /**
     * Specialization of @class NormalFluxDefinition.
     * Defines how to compute a LINEAR and HOMOGENEOUS normal flux.
     */
    template <class cfg>
    struct NormalFluxDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>> : NormalFluxDefinitionBase<cfg>
    {
        using field_t                           = typename cfg::input_field_t;
        using field_value_type                  = typename field_t::value_type;
        static constexpr std::size_t field_size = field_t::size;

        using flux_coeff_matrix_t   = CollapsMatrix<field_value_type, cfg::output_field_size, field_size>;
        using flux_stencil_coeffs_t = xt::xtensor_fixed<flux_coeff_matrix_t, xt::xshape<cfg::stencil_size>>;
        using cons_flux_func        = std::function<flux_stencil_coeffs_t(double)>;

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
        cons_flux_func cons_flux_function = nullptr;

        ~NormalFluxDefinition()
        {
            cons_flux_function = nullptr;
        }
    };

    /**
     * @class FluxDefinition:
     * Stores one object of @class NormalFluxDefinition for each positive Cartesian direction.
     */
    template <class cfg>
    class FluxDefinition
    {
      public:

        static constexpr std::size_t dim          = cfg::dim;
        static constexpr std::size_t stencil_size = cfg::stencil_size;
        using flux_computation_t                  = NormalFluxDefinition<cfg>;

      private:

        std::array<flux_computation_t, dim> m_normal_fluxes;

      public:

        FluxDefinition()
        {
            auto directions = positive_cartesian_directions<dim>();
            static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
                [&](auto integral_constant_d)
                {
                    static constexpr int d = decltype(integral_constant_d)::value;

                    DirectionVector<dim> direction = xt::view(directions, d);
                    m_normal_fluxes[d].direction   = direction;
                    m_normal_fluxes[d].stencil     = line_stencil_from<dim, d, stencil_size>(-static_cast<int>(stencil_size) / 2 + 1);
                    m_normal_fluxes[d].cons_flux_function = nullptr; // to be set by the user
                });
        }

        /**
         * This constructor sets the same flux function for all directions
         */
        explicit FluxDefinition(typename flux_computation_t::cons_flux_func flux_implem)
        {
            auto directions = positive_cartesian_directions<dim>();
            for (std::size_t d = 0; d < dim; ++d)
            {
                DirectionVector<dim> direction        = xt::view(directions, d);
                m_normal_fluxes[d].direction          = direction;
                m_normal_fluxes[d].stencil            = line_stencil_from<dim, d, stencil_size>(-static_cast<int>(stencil_size) / 2 + 1);
                m_normal_fluxes[d].cons_flux_function = flux_implem;
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

    template <class cfg>
    using FluxValue = typename NormalFluxDefinition<cfg>::flux_value_t;

    template <class cfg>
    using FluxValuePair = typename NormalFluxDefinition<cfg>::flux_value_pair_t;

    template <class cfg>
    using FluxStencilCoeffs = typename NormalFluxDefinition<cfg>::flux_stencil_coeffs_t;

} // end namespace samurai
