#pragma once
#include "../../interface.hpp"
#include "../explicit_scheme.hpp"
#include "flux_definition.hpp"

namespace samurai
{
    /**
     * @class FluxBasedSchemeDefinition
     * Contains:
     * - how the flux of the field is computed
     * - how the flux contributes to the scheme
     */
    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    class FluxBasedSchemeDefinition
    {
      public:

        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;

        using flux_computation_t = LinearNormalFluxDefinition<Field, stencil_size>;
        using field_value_type   = typename Field::value_type; // double
        using coeff_matrix_t     = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;
        using cell_coeffs_t      = xt::xtensor_fixed<coeff_matrix_t, xt::xshape<stencil_size>>;
        using flux_coeffs_t      = typename flux_computation_t::flux_coeffs_t;
        using cell_coeffs_func_t = std::function<cell_coeffs_t(flux_coeffs_t&, double, double)>;

      private:

        flux_computation_t m_flux;
        cell_coeffs_func_t m_contribution_func;
        cell_coeffs_func_t m_contribution_opposite_direction_func = nullptr;

      public:

        auto& flux()
        {
            return m_flux;
        }

        auto& contribution_func()
        {
            return m_contribution_func;
        }

        auto& contribution_opposite_direction_func()
        {
            return m_contribution_opposite_direction_func;
        }

        void set_flux(const flux_computation_t& flux)
        {
            m_flux = flux;
        }

        void set_contribution(cell_coeffs_func_t contribution_func)
        {
            m_contribution_func = contribution_func;
            if (!m_contribution_opposite_direction_func)
            {
                m_contribution_opposite_direction_func = contribution_func;
            }
        }

        void set_contribution_opposite_direction(cell_coeffs_func_t contribution_func)
        {
            m_contribution_opposite_direction_func = contribution_func;
        }

        /**
         * Computes and returns the contribution coefficients
         */
        cell_coeffs_t contribution(flux_coeffs_t& flux, double h_face, double h_cell)
        {
            return m_contribution_func(flux, h_face, h_cell);
        }

        cell_coeffs_t contribution_opposite_direction(flux_coeffs_t& flux, double h_face, double h_cell)
        {
            return m_contribution_opposite_direction_func(flux, h_face, h_cell);
        }
    };

    template <std::size_t output_field_size_, std::size_t stencil_size_, bool is_linear_ = true, bool is_heterogeneous_ = false>
    struct FluxBasedSchemeConfig
    {
        static constexpr std::size_t output_field_size = output_field_size_;
        static constexpr std::size_t stencil_size      = stencil_size_;
        static constexpr bool is_linear                = is_linear_;
        static constexpr bool is_heterogeneous         = is_heterogeneous_;
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

    /**
     * is_FluxBasedScheme
     */
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
