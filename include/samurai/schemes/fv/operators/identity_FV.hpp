#pragma once
#include "../cell_based_scheme.hpp"

namespace samurai
{
    template <class Field>
    auto make_identity()
    {
        static constexpr std::size_t dim = Field::dim;

        using cfg      = OneCellStencilFV<Field::size>;
        using bdry_cfg = BoundaryConfigFV<1>;

        CellBasedScheme<cfg, bdry_cfg, Field> identity;

        using local_matrix_t = typename decltype(identity)::local_matrix_t;

        identity.set_name("Identity");
        identity.stencil()           = center_only_stencil<dim>();
        identity.coefficients_func() = [](double) -> std::array<local_matrix_t, 1>
        {
            return {eye<local_matrix_t>()};
        };
        identity.is_symmetric(true);
        identity.is_spd(true);
        return identity;
    }

    template <class Field>
    [[deprecated("Use make_identity() instead.")]] auto make_identity_FV()
    {
        return make_identity<Field>();
    }

} // end namespace samurai
