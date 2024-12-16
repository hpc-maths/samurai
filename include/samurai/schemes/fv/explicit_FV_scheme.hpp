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
        using size_type      = typename scheme_t::size_type;

        static constexpr std::size_t dim = input_field_t::mesh_t::dim;

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

      protected:

        output_field_t create_output_field(input_field_t& input_field) const
        {
            output_field_t output_field(scheme().name() + "(" + input_field.name() + ")", input_field.mesh());
            output_field.fill(0);
            return output_field;
        }

      public:

        auto apply_to(input_field_t& input_field) const
        {
            output_field_t output_field = create_output_field(input_field);

            // update_bc(input_field);
            apply(output_field, input_field);

            return output_field;
        }

        auto apply_to(std::size_t d, input_field_t& input_field) const
        {
            output_field_t output_field = create_output_field(input_field);

            // update_bc(input_field);
            apply(d, output_field, input_field);

            return output_field;
        }

        virtual void apply(output_field_t& output_field, input_field_t& input_field) const
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                apply(d, output_field, input_field);
            }
        }

        virtual void apply(std::size_t /* d */, output_field_t& /* output_field */, input_field_t& /* input_field */) const
        {
            std::cerr << "The scheme '" << scheme().name() << "' cannot be applied by direction." << std::endl;
            assert(false);
            exit(EXIT_FAILURE);
        }
    };
}
