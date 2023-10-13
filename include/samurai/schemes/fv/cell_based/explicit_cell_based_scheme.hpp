#pragma once
#include "../explicit_FV_scheme.hpp"
#include "cell_based_scheme__lin_hom.hpp"
#include "cell_based_scheme__nonlin.hpp"

namespace samurai
{
    /**
     * LINEAR and HOMOGENEOUS explicit schemes
     */
    template <class cfg, class bdry_cfg>
    class Explicit<CellBasedScheme<cfg, bdry_cfg>, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>>
        : public ExplicitFVScheme<CellBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<CellBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;

        using base_class::scheme;

        static constexpr std::size_t field_size        = input_field_t::size;
        static constexpr std::size_t output_field_size = cfg::output_field_size;
        static constexpr std::size_t stencil_size      = cfg::scheme_stencil_size;
        static constexpr std::size_t center_index      = cfg::center_index;

      public:

        explicit Explicit(const scheme_t& scheme)
            : base_class(scheme)
        {
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const override
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
                                double coeff = this->scheme().cell_coeff(coeffs, c, field_i, field_j);
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
        : public ExplicitFVScheme<CellBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<CellBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;

        using base_class::scheme;

        static constexpr std::size_t output_field_size = cfg::output_field_size;

      public:

        explicit Explicit(const scheme_t& scheme)
            : base_class(scheme)
        {
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const override
        {
            scheme().for_each_stencil_center(
                input_field,
                [&](const auto& stencil_center, auto& contrib)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        field_value(output_field, stencil_center, field_i) += this->scheme().contrib_cmpnent(contrib, field_i);
                    }
                });
        }
    };
} // end namespace samurai
