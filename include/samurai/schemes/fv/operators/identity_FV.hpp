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

        explicit IdentityFV(Field& unknown)
            : base_class(unknown)
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

        bool matrix_is_symmetric(const Field& unknown) const override
        {
            return is_uniform(unknown.mesh());
        }

        bool matrix_is_spd(const Field& unknown) const override
        {
            return matrix_is_symmetric(unknown);
        }
    };

    template <class Field>
    [[deprecated("Use make_identity() instead.")]] auto make_identity_FV(Field& f)
    {
        return make_identity(f);
    }

    template <class Field>
    auto make_identity(Field& f)
    {
        return IdentityFV<Field>(f);
    }

} // end namespace samurai
