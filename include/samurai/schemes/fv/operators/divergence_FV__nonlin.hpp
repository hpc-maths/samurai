#pragma once
#include "../flux_based_scheme__nonlin.hpp"

namespace samurai
{

    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    auto make_divergence(const FluxDefinition<FluxType::NonLinear, Field, output_field_size, stencil_size>& flux_definition)
    {
        make_flux_based_scheme(flux_definition);
    }

} // end namespace samurai
