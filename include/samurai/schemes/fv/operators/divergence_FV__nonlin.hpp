#pragma once
#include "../flux_based_scheme__nonlin.hpp"

namespace samurai
{

    template <class cfg, class Field>
    auto make_divergence(const FluxDefinition<cfg, Field>& flux_definition)
    {
        return make_flux_based_scheme(flux_definition);
    }

} // end namespace samurai
