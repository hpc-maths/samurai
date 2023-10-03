#pragma once
#include "cell_based_scheme.hpp"

namespace samurai
{
    template <class cfg, class bdry_cfg>
    class CellBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
        : public FVScheme<typename cfg::input_field_t, cfg::output_field_size, bdry_cfg>
    {
      protected:

        using base_class = FVScheme<typename cfg::input_field_t, cfg::output_field_size, bdry_cfg>;
        using base_class::dim;
        using base_class::field_size;

      public:

        using cfg_t            = cfg;
        using bdry_cfg_t       = bdry_cfg;
        using field_t          = typename cfg::input_field_t;
        using mesh_t           = typename field_t::mesh_t;
        using field_value_type = typename field_t::value_type;

        using scheme_definition_t = CellBasedSchemeDefinition<cfg>;
        using scheme_stencil_t    = typename scheme_definition_t::scheme_stencil_t;
        using stencil_cells_t     = typename scheme_definition_t::stencil_cells_t;
        using scheme_value_t      = typename scheme_definition_t::scheme_value_t;
        using scheme_func         = typename scheme_definition_t::scheme_func;

      private:

        scheme_definition_t m_scheme_definition;

      public:

        explicit CellBasedScheme(const scheme_definition_t& scheme_definition)
            : m_scheme_definition(scheme_definition)
        {
        }

        CellBasedScheme() = default;

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

        void set_scheme_function(scheme_func scheme_function)
        {
            m_scheme_definition.scheme_function = scheme_function;
        }

        auto contribution(stencil_cells_t& stencil_cells, field_t& field) const
        {
            return m_scheme_definition.scheme_function(stencil_cells, field);
        }

        auto operator()(field_t& field)
        {
            auto explicit_scheme = make_explicit(*this);
            return explicit_scheme.apply_to(field);
        }

        inline static field_value_type cell_coeff(const scheme_value_t& coeffs, [[maybe_unused]] std::size_t field_i)
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
        void for_each_stencil_center(field_t& field, Func&& apply_contrib) const
        {
            for_each_stencil(field.mesh(),
                             stencil(),
                             [&](auto& stencil_cells)
                             {
                                 auto contrib = contribution(stencil_cells, field);
                                 apply_contrib(stencil_cells[cfg::center_index], contrib);
                             });
        }

        /**
         * This function is used in the Assembly class to iterate over the stencils
         * and receive the Jacobian coefficients.
         */
        template <class Func>
        void for_each_stencil_and_coeffs(const mesh_t& mesh, Func&& apply_jacobian_coeffs) const
        {
            for_each_stencil(mesh,
                             stencil(),
                             [&](auto& stencil_cells)
                             {
                                 auto contrib = contribution(stencil_cells, field);
                                 apply_jacobian_coeffs(stencil_cells, contrib);
                             });
        }
    };

} // end namespace samurai
