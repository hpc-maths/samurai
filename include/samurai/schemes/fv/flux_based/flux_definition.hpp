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

        using flux_value_t       = xt::xtensor_fixed<field_value_type, xt::xshape<cfg::output_field_size>>;
        using flux_func          = std::function<flux_value_t(stencil_cells_t&, field_t&)>;
        using opposite_flux_func = std::function<flux_value_t(flux_value_t&, stencil_cells_t&, field_t&)>;

        // using flux_jac_t         = CollapsMatrix<field_value_type, cfg::output_field_size, field_size>;
        // using flux_jacobian_func = std::function<flux_jac_t(stencil_cells_t&, field_t&)>;

        flux_func flux_function                   = nullptr;
        opposite_flux_func opposite_flux_function = [](auto& flux_value, auto&, auto&) -> flux_value_t
        {
            return -flux_value;
        };

        // flux_jacobian_func flux_jac_function = nullptr;

        ~NormalFluxDefinition()
        {
            flux_function = nullptr;
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
        using flux_func             = std::function<flux_stencil_coeffs_t(stencil_cells_t&)>;

        flux_func flux_function = nullptr;

        ~NormalFluxDefinition()
        {
            flux_function = nullptr;
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
        using flux_func             = std::function<flux_stencil_coeffs_t(double)>;

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
        flux_func flux_function = nullptr;

        ~NormalFluxDefinition()
        {
            flux_function = nullptr;
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
        using cfg_stencil2                        = FluxConfig<cfg::scheme_type, cfg::output_field_size, 2, typename cfg::input_field_t>;
        using flux_computation_t                  = NormalFluxDefinition<cfg>;
        using flux_computation_stencil2_t         = NormalFluxDefinition<cfg_stencil2>;

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
        explicit FluxDefinition(typename flux_computation_stencil2_t::flux_func flux_implem)
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

    template <class cfg>
    using FluxValue = typename NormalFluxDefinition<cfg>::flux_value_t;

    template <class cfg>
    using FluxStencilCoeffs = typename NormalFluxDefinition<cfg>::flux_stencil_coeffs_t;

} // end namespace samurai
