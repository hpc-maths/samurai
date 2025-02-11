#pragma once
#include "../explicit_FV_scheme.hpp"
#include "flux_based_scheme__nonlin.hpp"

namespace samurai
{
    /**
     * NON-LINEAR explicit schemes
     */
    template <class cfg, class bdry_cfg>
    class Explicit<FluxBasedScheme<cfg, bdry_cfg>, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
        : public ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;
        using size_type      = typename base_class::size_type;
        using base_class::scheme;

        static constexpr size_type output_field_size = scheme_t::output_field_size;

      public:

        using base_class::apply;

        explicit Explicit(scheme_t& s)
            : base_class(s)
        {
        }

      private:

        template <bool enable_max_level_flux>
        void _apply(std::size_t d, output_field_t& output_field, input_field_t& input_field)
        {
            // Interior interfaces
            scheme().template for_each_interior_interface<Run::Parallel, enable_max_level_flux>( // We need the 'template' keyword...
                d,
                input_field,
                [&](const auto& cell, auto& contrib)
                {
                    for (size_type field_i = 0; field_i < output_field_size; ++field_i)
                    {
                    // clang-format off
                        #pragma omp atomic update
                        field_value(output_field, cell, field_i) += this->scheme().flux_value_cmpnent(contrib, field_i);
                        // clang-format on
                    }
                });

            // Boundary interfaces
            if (scheme().include_boundary_fluxes())
            {
                scheme().template for_each_boundary_interface<Run::Parallel, enable_max_level_flux>( // We need the 'template' keyword...
                    d,
                    input_field,
                    [&](const auto& cell, auto& contrib)
                    {
                        for (size_type field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            field_value(output_field, cell, field_i) += this->scheme().flux_value_cmpnent(contrib, field_i);
                        }
                    });
            }
        }

      public:

        void apply(std::size_t d, output_field_t& output_field, input_field_t& input_field) override
        {
            if (args::enable_max_level_flux || scheme().enable_max_level_flux()) // cppcheck-suppress knownConditionTrueFalse
            {
                _apply<true>(d, output_field, input_field);
            }
            else
            {
                _apply<false>(d, output_field, input_field);
            }
        }
    };
} // end namespace samurai
