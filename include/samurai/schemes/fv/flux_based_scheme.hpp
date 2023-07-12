#pragma once
#include "../../interface.hpp"
#include "../explicit_scheme.hpp"
#include "FV_scheme.hpp"

namespace samurai
{
    /**
     * Defines how to compute a normal flux: e.g., Grad(u).n
     * - direction: e.g., right = {0, 1}
     * - stencil: e.g., current cell and right neighbour = {{0, 0}, {0, 1}}
     * - function returning the coefficients for the flux computation w.r.t. the stencil:
     *            auto get_flux_coeffs(double h)
     *            {
     *                // Grad(u).n = (u_1 - u_0)/h
     *                std::array<double, 2> flux_coeffs;
     *                flux_coeffs[0] = -1/h; // current cell
     *                flux_coeffs[1] =  1/h; // right neighbour
     *                return flux_coeffs;
     *            }
     */
    template <class Field, std::size_t stencil_size>
    struct NormalFluxComputation
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type;                      // double
        using flux_matrix_t = typename detail::LocalMatrix<field_value_type, 1, field_size>::Type; // 'double' if field_size = 1,
                                                                                                   // 'xtensor' representing a matrix
                                                                                                   // otherwise
        using flux_coeffs_t = std::array<flux_matrix_t, stencil_size>;

        DirectionVector<dim> direction;
        Stencil<stencil_size, dim> stencil;
        std::function<flux_coeffs_t(double)> get_flux_coeffs;
    };

    template <class Field, class Vector>
    auto normal_grad_order1(Vector& direction)
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;
        using flux_computation_t                = NormalFluxComputation<Field, 2>;
        using flux_matrix_t                     = typename flux_computation_t::flux_matrix_t;

        flux_computation_t normal_grad;
        normal_grad.direction       = direction;
        normal_grad.stencil         = in_out_stencil<dim>(direction);
        normal_grad.get_flux_coeffs = [](double h)
        {
            std::array<flux_matrix_t, 2> coeffs;
            if constexpr (field_size == 1)
            {
                coeffs[0] = -1 / h;
                coeffs[1] = 1 / h;
            }
            else
            {
                coeffs[0].fill(-1 / h);
                coeffs[1].fill(1 / h);
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

    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    struct FluxBasedCoefficients
    {
        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;

        using flux_computation_t = NormalFluxComputation<Field, stencil_size>;
        using field_value_type   = typename Field::value_type; // double
        using coeff_matrix_t     = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;
        using cell_coeffs_t      = std::array<coeff_matrix_t, stencil_size>;
        using flux_coeffs_t      = typename flux_computation_t::flux_coeffs_t; // std::array<flux_matrix_t, stencil_size>;
        using cell_coeffs_func_t = std::function<cell_coeffs_t(flux_coeffs_t&, double, double)>;

        flux_computation_t flux;
        cell_coeffs_func_t get_left_cell_coeffs;
        cell_coeffs_func_t get_right_cell_coeffs;
    };

    /**
     * Useful sizes to define the sparsity pattern of the matrix and perform the preallocation.
     */
    template <std::size_t output_field_size_, std::size_t stencil_size_>
    struct FluxBasedAssemblyConfig
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

        using coefficients_t = FluxBasedCoefficients<Field, output_field_size, stencil_size>;

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
