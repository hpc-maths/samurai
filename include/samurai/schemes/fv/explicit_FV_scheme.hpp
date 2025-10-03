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

        scheme_t* m_scheme = nullptr;

      public:

        explicit ExplicitFVScheme(scheme_t& scheme)
            : m_scheme(&scheme)
        {
        }

        virtual ~ExplicitFVScheme()
        {
        }

        auto& scheme()
        {
            return *m_scheme;
        }

        const auto& scheme() const
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

        auto apply_to(input_field_t& input_field)
        {
            output_field_t output_field = create_output_field(input_field);

            apply(output_field, input_field);

            return output_field;
        }

        auto apply_to(std::size_t d, input_field_t& input_field)
        {
            output_field_t output_field = create_output_field(input_field);

            apply(d, output_field, input_field);

            return output_field;
        }

        virtual void apply(output_field_t& output_field, input_field_t& input_field)
        {
            static constexpr int scheme_stencil_size = static_cast<int>(scheme_t::cfg_t::stencil_size);
            int mesh_stencil_size                    = input_field.mesh().cfg().max_stencil_size();

            if (scheme_stencil_size > mesh_stencil_size)
            {
                std::cerr << "The stencil size required by the scheme '" << scheme().name() << "' (" << scheme_stencil_size
                          << ") is larger than the max_stencil_size parameter of the mesh (" << mesh_stencil_size
                          << ").\nYou can set it with mesh_config.max_stencil_radius(" << scheme_stencil_size / 2
                          << ") or mesh_config.max_stencil_size(" << scheme_stencil_size << ")." << std::endl;
                exit(EXIT_FAILURE);
            }

            for (std::size_t d = 0; d < dim; ++d)
            {
                apply(d, output_field, input_field);
            }
        }

        virtual void apply(std::size_t /* d */, output_field_t& /* output_field */, input_field_t& /* input_field */)
        {
            std::cerr << "The scheme '" << scheme().name() << "' cannot be applied by direction." << std::endl;
            assert(false);
            exit(EXIT_FAILURE);
        }
    };
}
