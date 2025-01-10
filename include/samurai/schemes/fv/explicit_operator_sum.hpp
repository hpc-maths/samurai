#pragma once
#include "explicit_FV_scheme.hpp"
#include "scheme_operators.hpp"

namespace samurai
{
    template <class... Operators>
    class Explicit<OperatorSum<Operators...>> : public ExplicitFVScheme<OperatorSum<Operators...>>
    {
      public:

        using base_class = ExplicitFVScheme<OperatorSum<Operators...>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;
        using size_type      = typename base_class::size_type;
        using base_class::scheme;

        explicit Explicit(const scheme_t& sum_scheme)
            : base_class(sum_scheme)
        {
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const override
        {
            for_each(scheme().operators(),
                     [&](const auto& op)
                     {
                         op.apply(output_field, input_field);
                     });
        }

        void apply(std::size_t d, output_field_t& output_field, input_field_t& input_field) const override
        {
            for_each(scheme().operators(),
                     [&](const auto& op)
                     {
                         op.apply(d, output_field, input_field);
                     });
        }
    };

} // end namespace samurai
