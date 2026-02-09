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
        requires(cfg::scheme_type == SchemeType::LinearHomogeneous)
    class Explicit<CellBasedScheme<cfg, bdry_cfg>> : public ExplicitFVScheme<CellBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<CellBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;
        using size_type      = typename base_class::size_type;

        using base_class::scheme;

        static constexpr size_type input_n_comp   = input_field_t::n_comp;
        static constexpr size_type output_n_comp  = output_field_t::n_comp;
        static constexpr std::size_t stencil_size = cfg::stencil_size;
        static constexpr std::size_t center_index = cfg::center_index;

      public:

        using base_class::apply;

        explicit Explicit(scheme_t& s)
            : base_class(s)
        {
        }

        void apply(output_field_t& output_field, input_field_t& input_field) override
        {
            if constexpr (cfg::stencil_size > 1)
            {
                assert(input_field.ghosts_updated());
            }

            scheme().for_each_stencil_and_coeffs(
                input_field,
                [&](const auto& cells, const auto& coeffs)
                {
                    for (size_type field_i = 0; field_i < output_n_comp; ++field_i)
                    {
                        for (size_type field_j = 0; field_j < input_n_comp; ++field_j)
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
        requires(cfg::scheme_type == SchemeType::NonLinear)
    class Explicit<CellBasedScheme<cfg, bdry_cfg>> : public ExplicitFVScheme<CellBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<CellBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;
        using size_type      = typename base_class::size_type;

        using base_class::scheme;

        static constexpr size_type output_n_comp = output_field_t::n_comp;

      public:

        using base_class::apply;

        explicit Explicit(scheme_t& s)
            : base_class(s)
        {
        }

        void apply(output_field_t& output_field, input_field_t& input_field) override
        {
            scheme().for_each_stencil_center(
                input_field,
                [&](const auto& stencil_center, auto& contrib)
                {
                    for (size_type field_i = 0; field_i < output_n_comp; ++field_i)
                    {
                        field_value(output_field, stencil_center, field_i) += this->scheme().contrib_cmpnent(contrib, field_i);
                    }
                });
        }
    };
} // end namespace samurai
