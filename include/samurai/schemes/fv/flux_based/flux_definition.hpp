#pragma once
#include "../utils.hpp"
#include <functional>

namespace samurai
{
    template <SchemeType scheme_type_, std::size_t stencil_size_, class output_field_t_, class input_field_t_, class parameter_field_t_ = void*>
        requires(IsField<output_field_t_> && IsField<input_field_t_>)
    struct FluxConfig
    {
        static constexpr SchemeType scheme_type   = scheme_type_;
        static constexpr std::size_t stencil_size = stencil_size_;
        using output_field_t                      = std::decay_t<output_field_t_>;
        using input_field_t                       = std::decay_t<input_field_t_>;
        using parameter_field_t                   = std::decay_t<parameter_field_t_>;
        static constexpr bool has_parameter_field = !std::is_same_v<parameter_field_t, void*>; // cppcheck-suppress unusedStructMember
        static constexpr std::size_t dim          = input_field_t::dim;
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
        StencilAnalyzer<cfg::stencil_size, cfg::dim> stencil;
    };

    /**
     * @class NormalFluxDefinition defines how to compute a normal flux.
     * This struct inherits from @class NormalFluxDefinitionBase and is specialized for all flux types (see below).
     */
    template <class cfg, class enable = void>
    struct NormalFluxDefinition
    {
    };

    //----------------------------------//
    //                                  //
    //            Non-linear            //
    //                                  //
    //----------------------------------//

    template <class cfg>
    using FluxValue = CollapsFluxArray<typename cfg::input_field_t::value_type, cfg::output_field_t::n_comp, cfg::output_field_t::is_scalar>;

    template <class cfg>
    using FluxValuePair = StdArrayWrapper<FluxValue<cfg>, 2>;

    template <class cfg>
    using StencilJacobianPair = StdArrayWrapper<StencilJacobian<cfg>, 2>;

    template <class cfg>
    struct StencilData
    {
        StencilCells<cfg>& cells;
        double cell_length = 0;

        explicit StencilData(StencilCells<cfg>& c)
            : cells(c)
        {
        }
    };

    /**
     * Specialization of @class NormalFluxDefinition.
     * Defines how to compute a NON-LINEAR normal flux.
     */
    template <class cfg>
    struct NormalFluxDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>> : NormalFluxDefinitionBase<cfg>
    {
        using field_t = typename cfg::input_field_t;

        using flux_func = std::function<void(FluxValuePair<cfg>&, const StencilData<cfg>&, const StencilValues<cfg>&)>;  // non-conservative
        using cons_flux_func = std::function<void(FluxValue<cfg>&, const StencilData<cfg>&, const StencilValues<cfg>&)>; // conservative

        using jacobian_func      = std::function<StencilJacobianPair<cfg>(StencilCells<cfg>&, const field_t&)>; // non-conservative
        using cons_jacobian_func = std::function<StencilJacobian<cfg>(StencilCells<cfg>&, const field_t&)>;     // conservative

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

        cons_jacobian_func cons_jacobian_function = nullptr;
        jacobian_func jacobian_function           = nullptr;

        /**
         * @returns the non-conservative flux function that calls the conservative one.
         * This function is used to default 'flux_function' if it is not set.
         */
        flux_func flux_function_as_conservative() const
        {
            return [&](FluxValuePair<cfg>& fluxes, auto& data, const auto& field)
            {
                cons_flux_function(fluxes[0], data, field);
                fluxes[1] = -fluxes[0];
            };
        }

        /**
         * @returns the non-conservative Jacobian function that calls the conservative one.
         * This function is used to default 'jacobian_function' if it is not set.
         */
        jacobian_func jacobian_function_as_conservative() const
        {
            if (!cons_jacobian_function)
            {
                return nullptr;
            }

            return [&](auto& cells, const auto& field)
            {
                StencilJacobianPair<cfg> jacobians;
                jacobians[0] = cons_jacobian_function(cells, field);
                jacobians[1] = -jacobians[0];
                return jacobians;
            };
        }

        ~NormalFluxDefinition()
        {
            cons_flux_function = nullptr;
            flux_function      = nullptr;

            cons_jacobian_function = nullptr;
            jacobian_function      = nullptr;
        }
    };

    //----------------------------------//
    //                                  //
    //      Linear heterogeneous        //
    //                                  //
    //----------------------------------//

    template <class cfg>
    using FluxStencilCoeffs = StencilJacobian<cfg>;

    /**
     * Specialization of @class NormalFluxDefinition.
     * Defines how to compute a LINEAR and HETEROGENEOUS normal flux.
     */
    template <class cfg>
    struct NormalFluxDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHeterogeneous>> : NormalFluxDefinitionBase<cfg>
    {
        using cons_flux_func = std::function<FluxStencilCoeffs<cfg>(StencilCells<cfg>&)>;

        cons_flux_func cons_flux_function = nullptr;

        ~NormalFluxDefinition()
        {
            cons_flux_function = nullptr;
        }
    };

    //----------------------------------//
    //                                  //
    //       Linear homogeneous         //
    //                                  //
    //----------------------------------//

    /**
     * Specialization of @class NormalFluxDefinition.
     * Defines how to compute a LINEAR and HOMOGENEOUS normal flux.
     */
    template <class cfg>
    struct NormalFluxDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>> : NormalFluxDefinitionBase<cfg>
    {
        using cons_flux_func = std::function<FluxStencilCoeffs<cfg>(double)>;

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

    //----------------------------------//
    //                                  //
    //         Flux definition          //
    //                                  //
    //----------------------------------//

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
            set_default(nullptr);
        }

        /**
         * This constructor sets the same flux function for all directions
         */
        explicit FluxDefinition(typename flux_computation_t::cons_flux_func flux_implem)
        {
            set_default(flux_implem);
        }

      private:

        void set_default(typename flux_computation_t::cons_flux_func flux_implem)
        {
            auto directions = positive_cartesian_directions<dim>();
            static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
                [&](auto _d)
                {
                    static constexpr std::size_t d = _d();
                    DirectionVector<dim> direction = xt::view(directions, d);
                    m_normal_fluxes[d].direction   = direction;
                    m_normal_fluxes[d].stencil     = line_stencil_from<dim, d, stencil_size>(-static_cast<int>(stencil_size) / 2 + 1);
                    m_normal_fluxes[d].cons_flux_function = flux_implem;
                });
        }

      public:

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

} // end namespace samurai
