#pragma once
#include "../cell_based_scheme.hpp"

namespace samurai
{
    template <class Field, class cfg = OneCellStencilFV<Field::size>, class bdry_cfg = BoundaryConfigFV<1>>
    class IdentityFV : public CellBasedScheme<IdentityFV<Field>, cfg, bdry_cfg, Field>
    {
        using base_class = CellBasedScheme<IdentityFV<Field>, cfg, bdry_cfg, Field>;
        using base_class::dim;
        using local_matrix_t = typename base_class::local_matrix_t;

      public:

        IdentityFV()
        {
            this->set_name("Identity");
        }

        static constexpr auto stencil()
        {
            return center_only_stencil<dim>();
        }

        static std::array<local_matrix_t, 1> coefficients(double)
        {
            return {eye<local_matrix_t>()};
        }
    };

    template <class Field>
    auto make_identity()
    {
        IdentityFV<Field> id;
        id.is_symmetric(true);
        id.is_spd(true);
        return id;
    }

    template <class Field>
    [[deprecated("Use make_identity() instead.")]] auto make_identity_FV()
    {
        return make_identity<Field>();
    }

} // end namespace samurai
