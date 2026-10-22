#pragma once
#include "../explicit_FV_scheme.hpp"
#include "flux_based_scheme__nonlin.hpp"

#include <stdexcept>

#include <fmt/format.h>

namespace samurai
{
    /**
     * NON-LINEAR explicit schemes
     */
    template <class cfg, class bdry_cfg>
        requires(cfg::scheme_type == SchemeType::NonLinear)
    class Explicit<FluxBasedScheme<cfg, bdry_cfg>> : public ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;
        using size_type      = typename base_class::size_type;
        using base_class::scheme;

        static constexpr size_type output_n_comp = scheme_t::output_n_comp;

        // The fluxes at finer levels predict stencil_size / 2 values on each side of the face, which needs an
        // even stencil size, unless the stencil values are copied (prediction stencil radius 0, at most 4 cells).
        // See compute_stencil_values() in flux_based_scheme__nonlin.hpp.
        static constexpr bool finer_level_flux_supported = cfg::stencil_size % 2 == 0
                                                        || (input_field_t::mesh_t::config_t::prediction_stencil_radius == 0
                                                            && cfg::stencil_size <= 4);

      public:

        using base_class::apply;

        explicit Explicit(scheme_t& s)
            : base_class(s)
        {
        }

      private:

        template <bool enable_finer_level_flux>
        void _apply(std::size_t d, output_field_t& output_field, input_field_t& input_field)
        {
            assert(input_field.ghosts_updated());

            // Interior interfaces
            scheme().template for_each_interior_interface<Run::Parallel, enable_finer_level_flux>( // We need the 'template' keyword...
                d,
                input_field,
                [&](const auto& cell, auto& contrib)
                {
                    for (size_type field_i = 0; field_i < output_n_comp; ++field_i)
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
                scheme().template for_each_boundary_interface<Run::Parallel, enable_finer_level_flux>( // We need the 'template' keyword...
                    d,
                    input_field,
                    [&](const auto& cell, auto& contrib)
                    {
                        for (size_type field_i = 0; field_i < output_n_comp; ++field_i)
                        {
                            field_value(output_field, cell, field_i) += this->scheme().flux_value_cmpnent(contrib, field_i);
                        }
                    });
            }
        }

      public:

        void apply(std::size_t d, output_field_t& output_field, input_field_t& input_field) override
        {
            scheme().apply_directional_bc(input_field, d);

            if (args::finer_level_flux != 0 || scheme().enable_finer_level_flux()) // cppcheck-suppress knownConditionTrueFalse
            {
                // Instantiate the fluxes at finer levels only where they are implemented.
                if constexpr (finer_level_flux_supported)
                {
                    _apply<true>(d, output_field, input_field);
                }
                else
                {
                    throw std::runtime_error(fmt::format("The scheme '{}' has a stencil of odd size ({}): the fluxes at finer levels "
                                                         "(--finer-level-flux) are not implemented for odd stencil sizes.",
                                                         scheme().name(),
                                                         cfg::stencil_size));
                }
            }
            else
            {
                _apply<false>(d, output_field, input_field);
            }
        }
    };
} // end namespace samurai
