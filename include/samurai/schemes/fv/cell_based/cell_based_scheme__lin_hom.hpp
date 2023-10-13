#pragma once
#include "cell_based_scheme.hpp"

namespace samurai
{
    template <class cfg, class bdry_cfg>
    class CellBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>>
        : public FVScheme<CellBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>
    {
        using base_class = FVScheme<CellBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>;

      public:

        using base_class::dim;
        using base_class::field_size;
        using base_class::output_field_size;

        using cfg_t         = cfg;
        using bdry_cfg_t    = bdry_cfg;
        using input_field_t = typename base_class::input_field_t;
        using mesh_t        = typename base_class::mesh_t;

        using scheme_definition_t   = CellBasedSchemeDefinition<cfg>;
        using scheme_stencil_t      = typename scheme_definition_t::scheme_stencil_t;
        using stencil_coeffs_t      = typename scheme_definition_t::stencil_coeffs_t;
        using get_coefficients_func = typename scheme_definition_t::get_coefficients_func;

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

        const get_coefficients_func& coefficients_func() const
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

        inline double cell_coeff(const stencil_coeffs_t& coeffs,
                                 std::size_t cell_number_in_stencil,
                                 [[maybe_unused]] std::size_t field_i,
                                 [[maybe_unused]] std::size_t field_j) const
        {
            if constexpr (field_size == 1 && output_field_size == 1)
            {
                return coeffs[cell_number_in_stencil];
            }
            else
            {
                return coeffs[cell_number_in_stencil](field_i, field_j);
            }
        }

        template <class Func>
        void for_each_stencil_and_coeffs(input_field_t& field, Func&& apply_coeffs) const
        {
            auto& mesh      = field.mesh();
            auto stencil_it = make_stencil_iterator(mesh, stencil());

            for_each_level(mesh,
                           [&](std::size_t level)
                           {
                               auto coeffs = coefficients(cell_length(level));

                               for_each_stencil(mesh,
                                                level,
                                                stencil_it,
                                                [&](auto& stencil_cells)
                                                {
                                                    apply_coeffs(stencil_cells, coeffs);
                                                });
                           });
        }
    };

} // end namespace samurai
