#pragma once
#include "../../explicit_scheme.hpp"
#include "cell_based_scheme__lin_hom.hpp"
#include "cell_based_scheme__nonlin.hpp"

namespace samurai
{
    /**
     * LINEAR and HOMOGENEOUS explicit schemes
     */
    template <class cfg, class bdry_cfg>
    class Explicit<CellBasedScheme<cfg, bdry_cfg>, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>>
    {
        using scheme_t       = CellBasedScheme<cfg, bdry_cfg>;
        using input_field_t  = typename scheme_t::input_field_t;
        using output_field_t = typename scheme_t::output_field_t;

        static constexpr std::size_t field_size        = input_field_t::size;
        static constexpr std::size_t output_field_size = cfg::output_field_size;
        static constexpr std::size_t stencil_size      = cfg::scheme_stencil_size;
        static constexpr std::size_t center_index      = cfg::center_index;

      private:

        const scheme_t* m_scheme = nullptr;

      public:

        explicit Explicit(const scheme_t& scheme)
            : m_scheme(&scheme)
        {
        }

        auto& scheme() const
        {
            return *m_scheme;
        }

        auto apply_to(input_field_t& input_field) const
        {
            output_field_t output_field(scheme().name() + "(" + input_field.name() + ")", input_field.mesh());
            output_field.fill(0);

            update_bc(input_field);
            apply(output_field, input_field);

            return output_field;
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const
        {
            scheme().for_each_stencil_and_coeffs(
                input_field,
                [&](const auto& cells, const auto& coeffs)
                {
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                        {
                            for (std::size_t c = 0; c < stencil_size; ++c)
                            {
                                double coeff = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                field_value(output_field, cells[center_index], field_i) += coeff
                                                                                         * field_value(input_field, cells[c], field_j);
                            }
                        }
                    }
                });
        }
    };

    /**
     * NON-LINEAR explicit schemes
     */
    template <class cfg, class bdry_cfg>
    class Explicit<CellBasedScheme<cfg, bdry_cfg>, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
    {
        using scheme_t                                 = CellBasedScheme<cfg, bdry_cfg>;
        using input_field_t                            = typename scheme_t::input_field_t;
        using output_field_t                           = typename scheme_t::output_field_t;
        static constexpr std::size_t output_field_size = cfg::output_field_size;

      protected:

        const scheme_t* m_scheme = nullptr;

      public:

        explicit Explicit(const scheme_t& scheme)
            : m_scheme(&scheme)
        {
        }

        auto& scheme() const
        {
            return *m_scheme;
        }

        auto apply_to(input_field_t& input_field) const
        {
            output_field_t output_field(scheme().name() + "(" + input_field.name() + ")", input_field.mesh());
            output_field.fill(0);

            update_bc(input_field);
            apply(output_field, input_field);

            return output_field;
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const
        {
            scheme().for_each_stencil_center(
                input_field,
                [&](auto& stencil_center, auto& contrib)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        field_value(output_field, stencil_center, field_i) += scheme().contrib_cmpnent(contrib, field_i);
                    }
                });
        }
    };
} // end namespace samurai
