#pragma once
#include "cell_based/algebraic_operators.hpp"
#include "flux_based/algebraic_operators.hpp"

namespace samurai
{
    template <class... Operators>
    constexpr SchemeType scheme_type_of_sum()
    {
        constexpr bool
            one_is_nonlinear = std::disjunction_v<std::integral_constant<bool, Operators::cfg_t::scheme_type == SchemeType::NonLinear>...>;
        constexpr bool one_is_linear_het = std::disjunction_v<
            std::integral_constant<bool, Operators::cfg_t::scheme_type == SchemeType::LinearHeterogeneous>...>;

        if constexpr (one_is_nonlinear)
        {
            return SchemeType::NonLinear;
        }
        else if constexpr (one_is_linear_het)
        {
            return SchemeType::LinearHeterogeneous;
        }
        else
        {
            return SchemeType::LinearHomogeneous;
        }
    }

    template <class... Operators>
    constexpr std::size_t stencil_size_of_sum()
    {
        return std::max({Operators::cfg_t::stencil_size...});
    }

    template <class... Operators>
    constexpr std::size_t get_largest_stencil_index()
    {
        std::size_t max = std::max({Operators::cfg_t::stencil_size...});
        std::size_t i   = 0;
        for (const auto& size : {Operators::cfg_t::stencil_size...})
        {
            if (size == max)
            {
                break;
            }
            i++;
        }
        return i;
    }

    /**
     * @class OperatorSum:
     * Stores a list of operators that cannot be combined.
     * When an explicit execution of is requested, the operators are executed sequentially (see @class Explicit<OperatorSum>).
     * When a matrix assembly is requested for an implicit term, the operators add their coefficients sequentially
     * (see @class Assembly<OperatorSum>).
     */
    template <class... Operators>
    class OperatorSum
    {
      private:

        template <SchemeType scheme_type_, std::size_t stencil_size_, std::size_t output_field_size_>
        struct Config
        {
            static constexpr SchemeType scheme_type        = scheme_type_;
            static constexpr std::size_t stencil_size      = stencil_size_;
            static constexpr std::size_t output_field_size = output_field_size_;
        };

      public:

        using FirstOperatorType = std::tuple_element_t<0, std::tuple<Operators...>>;

        static constexpr std::size_t output_field_size = FirstOperatorType::cfg_t::output_field_size;
        using input_field_t                            = typename FirstOperatorType::input_field_t;
        using output_field_t                           = typename FirstOperatorType::output_field_t;
        using size_type                                = typename FirstOperatorType::size_type;
        using field_t                                  = input_field_t;

        using cfg_t = Config<scheme_type_of_sum<Operators...>(), stencil_size_of_sum<Operators...>(), output_field_size>;

        // cppcheck-suppress unusedStructMember
        static constexpr std::size_t largest_stencil_index = get_largest_stencil_index<Operators...>();

      private:

        std::tuple<Operators...> m_operators;

      public:

        explicit OperatorSum(const Operators&... operators)
            : m_operators(operators...)
        {
            static constexpr bool all_output_field_sizes_are_same = std::conjunction_v<
                std::integral_constant<bool, Operators::cfg_t::output_field_size == output_field_size>...>;
            static_assert(all_output_field_sizes_are_same, "Cannot add operators with different output field sizes.");
        }

        explicit OperatorSum(const std::tuple<Operators...>& operator_tuple)
            : m_operators(operator_tuple)
        {
            // static constexpr bool all_output_field_sizes_are_same = std::conjunction_v<
            //     std::integral_constant<bool, Operators::cfg_t::output_field_size == output_field_size>...>;
            // static_assert(all_output_field_sizes_are_same, "Cannot add operators with different output field sizes.");
        }

        auto& operators() const
        {
            return m_operators;
        }

        auto& operators()
        {
            return m_operators;
        }

        std::string name() const
        {
            std::stringstream ss;
            ss << "[ ";
            bool is_first = true;
            for_each(m_operators,
                     [&](const auto& op)
                     {
                         ss << (is_first ? "" : " + ") << op.name();
                         is_first = false;
                     });
            ss << " ]";
            return ss.str();
        }

        auto operator()(input_field_t& input_field) const
        {
            auto explicit_scheme = make_explicit(*this);
            return explicit_scheme.apply_to(input_field);
        }

        auto operator()(std::size_t d, input_field_t& input_field) const
        {
            auto explicit_scheme = make_explicit(*this);
            return explicit_scheme.apply_to(d, input_field);
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const
        {
            auto explicit_scheme = make_explicit(*this);
            explicit_scheme.apply(output_field, input_field);
        }
    };

    template <class... Operators>
    auto make_operator_sum(const Operators&... operators)
    {
        return OperatorSum<Operators...>(operators...);
    }

    template <class... Operators>
    auto make_operator_sum(const std::tuple<Operators...>& operator_tuple)
    {
        return OperatorSum<Operators...>(operator_tuple);
    }

    /**
     * Operator '+' between FluxBasedScheme and CellBasedScheme
     */

    // 2 uncombinable FluxBasedSchemes
    template <class cfg1,
              class bdry_cfg1,
              class cfg2,
              class bdry_cfg2,
              std::enable_if_t<!std::is_same_v<cfg1, cfg2> || !std::is_same_v<bdry_cfg1, bdry_cfg2>, bool> = true>
    auto operator+(const FluxBasedScheme<cfg1, bdry_cfg1>& scheme1, const FluxBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, scheme2);
    }

    // 2 uncombinable CellBasedSchemes
    template <class cfg1,
              class bdry_cfg1,
              class cfg2,
              class bdry_cfg2,
              std::enable_if_t<!std::is_same_v<cfg1, cfg2> || !std::is_same_v<bdry_cfg1, bdry_cfg2>, bool> = true>
    auto operator+(const CellBasedScheme<cfg1, bdry_cfg1>& scheme1, const CellBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, scheme2);
    }

    // FluxBasedScheme + CellBasedScheme
    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    auto operator+(const FluxBasedScheme<cfg1, bdry_cfg1>& scheme1, const CellBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, scheme2);
    }

    // CellBasedScheme + FluxBasedScheme
    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    auto operator+(const CellBasedScheme<cfg1, bdry_cfg1>& scheme1, const FluxBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, scheme2);
    }

    /**
     * Operator '+' between OperatorSum and CellBasedScheme
     */

    template <class Scheme, class... Operators>
    constexpr bool scheme_is_combinable = std::disjunction_v<std::is_same<Operators, Scheme>...>;

    // OperatorSum + combinable CellBasedScheme
    template <class cfg, class bdry_cfg, class... Operators>
    auto operator+(OperatorSum<Operators...>& sum_scheme, const CellBasedScheme<cfg, bdry_cfg>& scheme)
        -> std::enable_if_t<scheme_is_combinable<CellBasedScheme<cfg, bdry_cfg>, Operators...>, OperatorSum<Operators...>>
    {
        for_each(sum_scheme.operators(),
                 [&](auto& op)
                 {
                     if constexpr (std::is_same_v<CellBasedScheme<cfg, bdry_cfg>, std::decay_t<decltype(op)>>)
                     {
                         op = op + scheme;
                     }
                 });
        return sum_scheme;
    }

    // combinable CellBasedScheme + OperatorSum
    template <class cfg, class bdry_cfg, class... Operators>
    auto operator+(const CellBasedScheme<cfg, bdry_cfg>& scheme, OperatorSum<Operators...>& sum_scheme)
        -> std::enable_if_t<scheme_is_combinable<CellBasedScheme<cfg, bdry_cfg>, Operators...>, OperatorSum<Operators...>>
    {
        return sum_scheme + scheme;
    }

    // OperatorSum + uncombinable CellBasedScheme
    template <class cfg, class bdry_cfg, class... Operators>
    auto operator+(const OperatorSum<Operators...>& sum_scheme, const CellBasedScheme<cfg, bdry_cfg>& scheme)
        -> std::enable_if_t<!scheme_is_combinable<CellBasedScheme<cfg, bdry_cfg>, Operators...>,
                            OperatorSum<Operators..., CellBasedScheme<cfg, bdry_cfg>>>
    {
        auto added = std::tuple_cat(sum_scheme.operators(), std::make_tuple(scheme));
        return make_operator_sum(added);
    }

    // uncombinable CellBasedScheme + OperatorSum
    template <class cfg, class bdry_cfg, class... Operators>
    auto operator+(const CellBasedScheme<cfg, bdry_cfg>& scheme, const OperatorSum<Operators...>& sum_scheme)
        -> std::enable_if_t<!scheme_is_combinable<CellBasedScheme<cfg, bdry_cfg>, Operators...>,
                            OperatorSum<CellBasedScheme<cfg, bdry_cfg>, Operators...>>
    {
        auto added = std::tuple_cat(std::make_tuple(scheme), sum_scheme.operators());
        return make_operator_sum(added);
    }

    /**
     * Operator '-' between FluxBasedScheme and CellBasedScheme
     */
    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    auto operator-(const FluxBasedScheme<cfg1, bdry_cfg1>& scheme1, const CellBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, -scheme2);
    }

    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    auto operator-(const CellBasedScheme<cfg1, bdry_cfg1>& scheme1, const FluxBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, -scheme2);
    }

    /**
     * Operator '-' between OperatorSum and CellBasedScheme
     */
    template <class cfg, class bdry_cfg, class... Operators>
    auto operator-(const OperatorSum<Operators...>& sum_scheme, const CellBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return sum_scheme + ((-1) * scheme);
    }

    template <class cfg, class bdry_cfg, class... Operators>
    auto operator-(const CellBasedScheme<cfg, bdry_cfg>& scheme, const OperatorSum<Operators...>& sum_scheme)
    {
        return scheme + ((-1) * sum_scheme);
    }

    /**
     * Operator '-' between OperatorSum and FluxBasedScheme
     */
    template <class cfg, class bdry_cfg, class... Operators>
    auto operator-(const OperatorSum<Operators...>& sum_scheme, const FluxBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return sum_scheme + ((-1) * scheme);
    }

    template <class cfg, class bdry_cfg, class... Operators>
    auto operator-(const FluxBasedScheme<cfg, bdry_cfg>& scheme, const OperatorSum<Operators...>& sum_scheme)
    {
        return scheme + ((-1) * sum_scheme);
    }

    /**
     * Multiplication by scalar
     */
    template <class... Operators>
    OperatorSum<Operators...> operator*(double scalar, OperatorSum<Operators...>&& sum_scheme)
    {
        for_each(sum_scheme.operators(),
                 [&](auto& op)
                 {
                     op = scalar * op;
                 });
        return sum_scheme;
    }

    template <class... Operators>
    OperatorSum<Operators...> operator*(double scalar, const OperatorSum<Operators...>& sum_scheme)
    {
        auto result = sum_scheme;
        for_each(result.operators(),
                 [&](auto& op)
                 {
                     op = scalar * op;
                 });
        return result;
    }

} // end namespace samurai
