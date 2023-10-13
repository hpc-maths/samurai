#pragma once
#include "../explicit_scheme.hpp"
#include "FV_scheme.hpp"

namespace samurai
{
    /**
     * This is the base class of @class Explicit<CellBasedScheme> and @class Explicit<FluxBasedScheme>.
     */
    template <class Scheme>
    class ExplicitFVScheme
    {
      public:

        using scheme_t       = Scheme;
        using input_field_t  = typename scheme_t::input_field_t;
        using output_field_t = typename scheme_t::output_field_t;

      private:

        const scheme_t* m_scheme = nullptr;

      public:

        explicit ExplicitFVScheme(const scheme_t& scheme)
            : m_scheme(&scheme)
        {
        }

        virtual ~ExplicitFVScheme()
        {
        }

        auto& scheme() const
        {
            return *m_scheme;
        }

        auto apply_to(input_field_t& input_field) const
        {
            output_field_t output_field(scheme().name() + "(" + input_field.name() + ")", input_field.mesh());
            output_field.fill(0);

            update_bc(input_field);
            apply(output_field, input_field);

            return output_field;
        }

        virtual void apply(output_field_t& output_field, input_field_t& input_field) const = 0;
    };
}
