#pragma once
#include "cell_based_scheme.hpp"

namespace samurai
{
    template <class cfg, class bdry_cfg>
    class CellBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
        : public FVScheme<CellBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>
    {
        using base_class = FVScheme<CellBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>;

      public:

        using base_class::dim;
        using base_class::field_size;
        using base_class::output_field_size;
        using size_type = typename base_class::size_type;

        using cfg_t            = cfg;
        using bdry_cfg_t       = bdry_cfg;
        using input_field_t    = typename base_class::input_field_t;
        using mesh_t           = typename base_class::mesh_t;
        using field_value_type = typename base_class::field_value_type;

        using scheme_definition_t = CellBasedSchemeDefinition<cfg>;
        using scheme_stencil_t    = typename scheme_definition_t::scheme_stencil_t;
        using stencil_cells_t     = typename scheme_definition_t::stencil_cells_t;
        using scheme_func         = typename scheme_definition_t::scheme_func;
        using local_scheme_func   = typename scheme_definition_t::local_scheme_func;
        using jacobian_func       = typename scheme_definition_t::jacobian_func;
        using local_jacobian_func = typename scheme_definition_t::local_jacobian_func;

      private:

        scheme_definition_t m_scheme_definition;

      public:

        explicit CellBasedScheme(const scheme_definition_t& scheme_definition)
            : m_scheme_definition(scheme_definition)
        {
        }

        CellBasedScheme() = default;

        auto& scheme_definition()
        {
            return m_scheme_definition;
        }

        auto& stencil() const
        {
            return m_scheme_definition.stencil;
        }

        auto& stencil()
        {
            return m_scheme_definition.stencil;
        }

        void set_stencil(const scheme_stencil_t& stencil)
        {
            m_scheme_definition.stencil = stencil;
        }

        void set_stencil(scheme_stencil_t&& stencil)
        {
            m_scheme_definition.stencil = stencil;
        }

        const scheme_func& scheme_function() const
        {
            return m_scheme_definition.scheme_function;
        }

        scheme_func& scheme_function()
        {
            return m_scheme_definition.scheme_function;
        }

        const local_scheme_func& local_scheme_function() const
        {
            return m_scheme_definition.local_scheme_function;
        }

        local_scheme_func& local_scheme_function()
        {
            return m_scheme_definition.local_scheme_function;
        }

        template <class scheme_function_t>
        void set_scheme_function(scheme_function_t scheme_function)
        {
            m_scheme_definition.scheme_function       = scheme_function;
            m_scheme_definition.local_scheme_function = scheme_function;
        }

        auto contribution(stencil_cells_t& stencil_cells, input_field_t& field) const
        {
            return m_scheme_definition.scheme_function(stencil_cells, field);
        }

        const jacobian_func& jacobian_function() const
        {
            return m_scheme_definition.jacobian_function;
        }

        jacobian_func& jacobian_function()
        {
            return m_scheme_definition.jacobian_function;
        }

        const local_jacobian_func& local_jacobian_function() const
        {
            return m_scheme_definition.local_jacobian_function;
        }

        local_jacobian_func& local_jacobian_function()
        {
            return m_scheme_definition.local_jacobian_function;
        }

        template <class jacobian_function_t>
        void set_jacobian_function(jacobian_function_t jacobian_function)
        {
            m_scheme_definition.jacobian_function       = jacobian_function;
            m_scheme_definition.local_jacobian_function = jacobian_function;
        }

        auto jacobian_coefficients(stencil_cells_t& stencil_cells, input_field_t& field) const
        {
            return m_scheme_definition.jacobian_function(stencil_cells, field);
        }

        inline field_value_type contrib_cmpnent(const SchemeValue<cfg>& coeffs, [[maybe_unused]] size_type field_i) const
        {
            if constexpr (cfg::output_field_size == 1)
            {
                return coeffs;
            }
            else
            {
                return coeffs(field_i);
            }
        }

        /**
         * This function is used in the Explicit class to iterate over the stencil centers
         * and receive the contribution computed from the stencil.
         */
        template <class Func>
        void for_each_stencil_center(input_field_t& field, Func&& apply_contrib) const
        {
            for_each_stencil(field.mesh(),
                             stencil(),
                             [&](auto& stencil_cells)
                             {
                                 if constexpr (cfg::stencil_size == 1)
                                 {
                                     auto contrib = contribution(stencil_cells[0], field);
                                     apply_contrib(stencil_cells[cfg::center_index], contrib);
                                 }
                                 else
                                 {
                                     auto contrib = contribution(stencil_cells, field);
                                     apply_contrib(stencil_cells[cfg::center_index], contrib);
                                 }
                             });
        }

        /**
         * This function is used in the Assembly class to iterate over the stencils
         * and receive the Jacobian coefficients.
         */
        template <class Func>
        void for_each_stencil_and_coeffs(input_field_t& field, Func&& apply_jacobian_coeffs) const
        {
            if (!jacobian_function())
            {
                std::cerr << "The jacobian function of operator '" << this->name() << "' has not been implemented." << std::endl;
                std::cerr << "Use option -snes_mf or -snes_fd for an automatic computation of the jacobian matrix." << std::endl;
                exit(EXIT_FAILURE);
            }

            for_each_stencil(field.mesh(),
                             stencil(),
                             [&](auto& stencil_cells)
                             {
                                 if constexpr (cfg::stencil_size == 1)
                                 {
                                     auto coeffs = jacobian_coefficients(stencil_cells[0], field);
                                     apply_jacobian_coeffs(stencil_cells, coeffs);
                                 }
                                 else
                                 {
                                     auto coeffs = jacobian_coefficients(stencil_cells, field);
                                     apply_jacobian_coeffs(stencil_cells, coeffs);
                                 }
                             });
        }
    };

} // end namespace samurai
