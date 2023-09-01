#pragma once
#include "../flux_based_scheme__nonlin.hpp"

namespace samurai
{
    template <class Field,
              std::size_t output_field_size,
              std::size_t stencil_size = 2,
              // scheme config
              std::size_t dim = Field::dim,
              class cfg       = FluxBasedSchemeConfig<output_field_size, stencil_size, false, true>,
              class bdry_cfg  = BoundaryConfigFV<stencil_size / 2>>
    class DivergenceFV_NonLin : public FluxBasedScheme<DivergenceFV_NonLin<Field, output_field_size, stencil_size>, cfg, bdry_cfg, Field>
    {
        using base_class = FluxBasedScheme<DivergenceFV_NonLin<Field, output_field_size, stencil_size>, cfg, bdry_cfg, Field>;

      public:

        using scheme_definition_t               = typename base_class::scheme_definition_t;
        using flux_definition_t                 = typename scheme_definition_t::flux_definition_t;
        using flux_value_t                      = typename scheme_definition_t::flux_value_t;
        using scheme_contrib_t                  = typename scheme_definition_t::scheme_contrib_t;
        static constexpr std::size_t field_size = Field::size;

        explicit DivergenceFV_NonLin(const flux_definition_t& flux_definition, Field& u)
            : base_class(flux_definition, u)
        {
            this->set_name("Divergence");
            // static_assert(field_size == dim, "The field put into the divergence operator must have a size equal to the space
            // dimension.");
            add_contribution_to_scheme_definition();
        }

      private:

        void add_contribution_to_scheme_definition()
        {
            static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
                [&](auto integral_constant_d)
                {
                    static constexpr int d = decltype(integral_constant_d)::value;
                    this->definition()[d].set_contribution(add_flux_to_col<d>);
                });
        }

        template <std::size_t d>
        static scheme_contrib_t add_flux_to_col(flux_value_t& flux)
        {
            static_assert(std::is_same_v<scheme_contrib_t, flux_value_t>);
            return flux;
        }
    };

    template <class Field>
    auto make_divergence_nonlin(Field& f)
    {
        static constexpr std::size_t output_field_size = 1;
        static constexpr std::size_t stencil_size      = 2;

        auto flux_definition = make_flux_definition<Field, output_field_size, stencil_size>(get_average_value<Field>);
        return make_divergence_FV(flux_definition, f);
    }

    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    auto make_divergence_FV(const FluxDefinition<Field, output_field_size, stencil_size, false, true>& flux_definition, Field& f)
    {
        return DivergenceFV_NonLin<Field, output_field_size, stencil_size>(flux_definition, f);
    }

} // end namespace samurai
