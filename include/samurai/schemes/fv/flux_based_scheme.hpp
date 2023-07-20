#pragma once
#include "../../interface.hpp"
#include "../explicit_scheme.hpp"
#include "FV_scheme.hpp"

namespace samurai
{
    /**
     * Defines how to compute a normal flux
     */
    template <class Field, std::size_t stencil_size>
    struct NormalFluxComputation
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
        using flux_computation_t                = NormalFluxComputation<Field, 2>;
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
        using flux_computation_t         = NormalFluxComputation<Field, 2>;

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
        using flux_computation_t                = NormalFluxComputation<Field, 2>;
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

    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    struct FluxBasedSchemeDefinition
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;

        using flux_computation_t = NormalFluxComputation<Field, stencil_size>;
        using field_value_type   = typename Field::value_type; // double
        using coeff_matrix_t     = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;
        using cell_coeffs_t      = xt::xtensor_fixed<coeff_matrix_t, xt::xshape<stencil_size>>;
        using flux_coeffs_t      = typename flux_computation_t::flux_coeffs_t;
        using cell_coeffs_func_t = std::function<cell_coeffs_t(flux_coeffs_t&, double, double)>;

        flux_computation_t flux;
        cell_coeffs_func_t contribution;
        cell_coeffs_func_t contribution_opposite_direction = nullptr;
    };

    template <std::size_t output_field_size_, std::size_t stencil_size_>
    struct FluxBasedSchemeConfig
    {
        static constexpr std::size_t output_field_size = output_field_size_;
        static constexpr std::size_t stencil_size      = stencil_size_;
    };

    template <class DerivedScheme, class cfg, class bdry_cfg, class Field>
    class FluxBasedScheme : public FVScheme<DerivedScheme, Field, cfg::output_field_size, bdry_cfg>
    {
      protected:

        using base_class = FVScheme<DerivedScheme, Field, cfg::output_field_size, bdry_cfg>;
        using base_class::dim;
        using base_class::field_size;
        using field_value_type = typename base_class::field_value_type;

      public:

        using cfg_t                                    = cfg;
        using bdry_cfg_t                               = bdry_cfg;
        using field_t                                  = Field;
        static constexpr std::size_t output_field_size = cfg::output_field_size;
        static constexpr std::size_t stencil_size      = cfg::stencil_size;

        using scheme_definition_t = FluxBasedSchemeDefinition<Field, output_field_size, stencil_size>;

        explicit FluxBasedScheme(Field& unknown)
            : base_class(unknown)
        {
        }

        auto operator()(Field& f)
        {
            auto explicit_scheme = make_explicit(this->derived_cast());
            return explicit_scheme.apply_to(f);
        }
    };

    template <class Scheme, typename = void>
    struct is_FluxBasedScheme : std::false_type
    {
    };

    template <class Scheme>
    struct is_FluxBasedScheme<
        Scheme,
        std::enable_if_t<
            std::is_base_of_v<FluxBasedScheme<Scheme, typename Scheme::cfg_t, typename Scheme::bdry_cfg_t, typename Scheme::field_t>, Scheme>>>
        : std::true_type
    {
    };

    template <class Scheme>
    inline constexpr bool is_FluxBasedScheme_v = is_FluxBasedScheme<Scheme>::value;

} // end namespace samurai
