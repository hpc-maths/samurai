#pragma once
#include "cell_based_scheme.hpp"

namespace samurai
{
    template <class cfg, class bdry_cfg>
    class CellBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>>
        : public FVScheme<typename cfg::input_field_t, cfg::output_field_size, bdry_cfg>
    {
      protected:

        using base_class = FVScheme<typename cfg::input_field_t, cfg::output_field_size, bdry_cfg>;
        using base_class::dim;
        using base_class::field_size;

      public:

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;
        using field_t    = typename cfg::input_field_t;

        using scheme_definition_t   = CellBasedSchemeDefinition<cfg>;
        using scheme_stencil_t      = typename scheme_definition_t::scheme_stencil_t;
        using get_coefficients_func = typename scheme_definition_t::get_coefficients_func;

      private:

        scheme_definition_t m_scheme_definition;

      public:

        explicit CellBasedScheme(const scheme_definition_t& scheme_definition)
            : m_scheme_definition(scheme_definition)
        {
        }

        CellBasedScheme()
        {
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

        get_coefficients_func& coefficients_func() const
        {
            return m_scheme_definition.get_coefficients_function;
        }

        get_coefficients_func& coefficients_func()
        {
            return m_scheme_definition.get_coefficients_function;
        }

        void set_coefficients_func(get_coefficients_func get_coefficients_function)
        {
            m_scheme_definition.get_coefficients_function = get_coefficients_function;
        }

        auto coefficients(double h) const
        {
            return m_scheme_definition.get_coefficients_function(h);
        }
    };

} // end namespace samurai
