#pragma once
#include "../cell_based_scheme.hpp"

namespace samurai
{
    template <class Field, class cfg = OneCellStencilFV<Field::size>, class bdry_cfg = BoundaryConfigFV<1>>
    class IdentityFV : public CellBasedScheme<cfg, bdry_cfg, Field>
    {
        using base_class = CellBasedScheme<cfg, bdry_cfg, Field>;
        using base_class::dim;
        using local_matrix_t = typename base_class::local_matrix_t;

      public:

        IdentityFV()
        {
            this->set_name("Identity");
            this->stencil()           = center_only_stencil<dim>();
            this->coefficients_func() = [](double) -> std::array<local_matrix_t, 1>
            {
                return {eye<local_matrix_t>()};
            };
            this->is_symmetric(true);
            this->is_spd(true);
        }
    };

    template <class Field>
    auto make_identity()
    {
        return IdentityFV<Field>();
    }

    template <class Field>
    [[deprecated("Use make_identity() instead.")]] auto make_identity_FV()
    {
        return make_identity<Field>();
    }

} // end namespace samurai
