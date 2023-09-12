#pragma once
#include "../flux_based_scheme__nonlin.hpp"

namespace samurai
{
    template <class Field,
              std::size_t output_field_size,
              std::size_t stencil_size = 2,
              // scheme config
              std::size_t dim = Field::dim,
              class cfg       = FluxBasedSchemeConfig<FluxType::NonLinear, output_field_size, stencil_size>,
              class bdry_cfg  = BoundaryConfigFV<stencil_size / 2>>
    class DivergenceFV_NonLin : public FluxBasedScheme<DivergenceFV_NonLin<Field, output_field_size, stencil_size>, cfg, bdry_cfg, Field>
    {
        using base_class = FluxBasedScheme<DivergenceFV_NonLin<Field, output_field_size, stencil_size>, cfg, bdry_cfg, Field>;

      public:

        using scheme_definition_t = typename base_class::scheme_definition_t;
        using flux_definition_t   = typename scheme_definition_t::flux_definition_t;

        explicit DivergenceFV_NonLin(const flux_definition_t& flux_definition)
            : base_class(flux_definition)
        {
            this->set_name("Flux divergence");
        }
    };

    template <class Field>
    auto make_divergence_nonlin()
    {
        static constexpr std::size_t output_field_size = 1;

        auto flux_definition = make_flux_definition<Field, output_field_size>(get_average_value<Field>);
        return make_divergence_FV(flux_definition);
    }

    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    auto make_divergence(const FluxDefinition<FluxType::NonLinear, Field, output_field_size, stencil_size>& flux_definition)
    {
        return DivergenceFV_NonLin<Field, output_field_size, stencil_size>(flux_definition);
    }

} // end namespace samurai
