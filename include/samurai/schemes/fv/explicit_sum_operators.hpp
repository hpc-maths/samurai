#pragma once
#include "../explicit_scheme.hpp"
#include "scheme_operators.hpp"

namespace samurai
{
    template <class... Operators>
    class Explicit<OperatorSum<Operators...>>
    {
      public:

        using scheme_t       = OperatorSum<Operators...>;
        using input_field_t  = typename scheme_t::input_field_t;
        using output_field_t = typename scheme_t::output_field_t;

        static constexpr std::size_t output_field_size = scheme_t::cfg_t::output_field_size;

      private:

        const scheme_t* m_sum_scheme = nullptr;

        // std::tuple<Explicit<Operators>...> m_explicit_ops;

      public:

        explicit Explicit(const scheme_t& sum_scheme)
            : m_sum_scheme(&sum_scheme)
        // , m_explicit_ops(transform(sum_scheme.operators(),
        //                            [](const auto& op)
        //                            {
        //                                return make_explicit(op);
        //                            }))
        {
        }

        auto& scheme() const
        {
            return *m_sum_scheme;
        }

        auto apply_to(input_field_t& input_field) const
        {
            output_field_t output_field(scheme().name() + "(" + input_field.name() + ")", input_field.mesh());
            // output_field_t output_field("f_newton", input_field.mesh());
            output_field.fill(0);

            update_bc(input_field);
            apply(output_field, input_field);

            return output_field;
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const
        {
            for_each(m_sum_scheme->operators(),
                     [&](const auto& op)
                     {
                         op.apply(output_field, input_field);
                     });
        }
    };

} // end namespace samurai
