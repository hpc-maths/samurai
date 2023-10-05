#pragma once
#include "../explicit_scheme.hpp"
#include "scheme_operators.hpp"

namespace samurai
{
    template <class... Operators>
    class Explicit<OperatorSum<Operators...>>
    {
      public:

        using scheme_t = OperatorSum<Operators...>;
        using field_t  = typename scheme_t::field_t;

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

        auto apply_to(field_t& f)
        {
            auto result = make_field<typename field_t::value_type, output_field_size, field_t::is_soa>(scheme().name() + "(" + f.name() + ")",
                                                                                                       f.mesh());
            result.fill(0);

            for_each(m_sum_scheme->operators(),
                     [&](const auto& op)
                     {
                         result = result + op(f);
                     });

            return result;
        }
    };

} // end namespace samurai
