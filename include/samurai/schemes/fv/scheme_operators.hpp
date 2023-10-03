#pragma once
#include "cell_based/algebraic_operators.hpp"
#include "flux_based/algebraic_operators.hpp"

namespace samurai
{
    /**
     * Addition of a flux-based scheme and a cell-based scheme.
     */
    template <class FluxScheme, class CellScheme>
    class FluxBasedScheme_Sum_CellBasedScheme
    {
        //   public:

        //     using field_t = typename FluxScheme::field_t;

        //   private:

        //     std::string m_name = "(unnamed)";
        //     FluxScheme m_flux_scheme;
        //     CellScheme m_cell_scheme;

        //   public:

        //     FluxBasedScheme_Sum_CellBasedScheme(const FluxScheme& flux_scheme, const CellScheme& cell_scheme)
        //         : m_flux_scheme(flux_scheme)
        //         , m_cell_scheme(cell_scheme)
        //     {
        //         this->m_name = m_flux_scheme.name() + " + " + m_cell_scheme.name();
        //     }

        //     std::string name() const
        //     {
        //         return m_name;
        //     }

        //     auto& flux_scheme() const
        //     {
        //         return m_flux_scheme;
        //     }

        //     auto& cell_scheme() const
        //     {
        //         return m_cell_scheme;
        //     }

        //     bool matrix_is_symmetric() const // override
        //     {
        //         return m_flux_scheme.matrix_is_symmetric() && m_cell_scheme.matrix_is_symmetric();
        //     }

        //     bool matrix_is_spd() const // override
        //     {
        //         return m_flux_scheme.matrix_is_spd() && m_cell_scheme.matrix_is_spd();
        //     }
    };

    // Operator +
    // template <typename FluxScheme,
    //           typename CellScheme,
    //           std::enable_if_t<is_CellBasedScheme_v<CellScheme> && is_FluxBasedScheme_v<FluxScheme>, bool> = true>
    // auto operator+(const FluxScheme& flux_scheme, const CellScheme& cell_scheme)
    // {
    //     return FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>(flux_scheme, cell_scheme);
    // }

    // Operator + with reference rvalue
    /*template <typename FluxScheme, typename CellScheme, std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true,
    std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true> auto operator + (FluxScheme&& flux_scheme, const CellScheme&
    cell_scheme)
    {
        //return flux_scheme + cell_scheme;
        return FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>(flux_scheme, cell_scheme);
    }*/

    // Operator + in the reverse order
    // template <typename CellScheme,
    //           typename FluxScheme,
    //           std::enable_if_t<is_CellBasedScheme_v<CellScheme> && is_FluxBasedScheme_v<FluxScheme>, bool> = true>
    // auto operator+(const CellScheme& cell_scheme, const FluxScheme& flux_scheme)
    // {
    //     return flux_scheme + cell_scheme;
    // }

    // Operator + in the reverse order with reference rvalue
    /*template <typename CellScheme, typename FluxScheme, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true,
    std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true> auto operator + (const CellScheme& cell_scheme, FluxScheme&&
    flux_scheme)
    {
        return flux_scheme + cell_scheme;
    }*/

    // Operator + with reference rvalue
    /*template <typename CellScheme, typename FluxScheme, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true,
    std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true> auto operator + (CellScheme&& cell_scheme, FluxScheme&& flux_scheme)
    {
        return flux_scheme + cell_scheme;
    }*/

    // FluxBasedScheme_Sum_CellBasedScheme + flux_scheme2
    // template <typename CellScheme, typename FluxScheme, typename FluxScheme2, std::enable_if_t<is_FluxBasedScheme_v<FluxScheme2>, bool> =
    // true> auto operator+(const FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>& sum_scheme, const FluxScheme2& flux_scheme2)
    // {
    //     return (sum_scheme.flux_scheme() + flux_scheme2) + sum_scheme.cell_scheme();
    // }

    // FluxBasedScheme_Sum_CellBasedScheme + cell_scheme2
    // template <typename CellScheme, typename FluxScheme, typename CellScheme2, std::enable_if_t<is_CellBasedScheme_v<CellScheme2>, bool> =
    // true> auto operator+(const FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>& sum_scheme, const CellScheme2& cell_scheme2)
    // {
    //     return (sum_scheme.cell_scheme() + cell_scheme2) + sum_scheme.flux_scheme();
    // }

    // FluxBasedScheme_Sum_CellBasedScheme - cell_scheme2
    // template <typename CellScheme, typename FluxScheme, typename CellScheme2, std::enable_if_t<is_CellBasedScheme_v<CellScheme2>, bool> =
    // true> auto operator-(const FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>& sum_scheme, const CellScheme2& cell_scheme2)
    // {
    //     return (sum_scheme.cell_scheme() - cell_scheme2) + sum_scheme.flux_scheme();
    // }

    // Sum_CellBasedScheme + flux_scheme
    // template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2, class cfg3, class bdry_cfg3>
    // auto
    // operator+(const Sum_CellBasedScheme<cfg1, bdry_cfg1, cfg2, bdry_cfg2>& sum_scheme, const FluxBasedScheme<cfg3, bdry_cfg3>&
    // flux_scheme)
    // {
    //     return (sum_scheme.cell_scheme() + cell_scheme2) + sum_scheme.flux_scheme();
    // }

    template <class... Operators>
    class OperatorSum
    {
      private:

        using FirstOperatorType = std::tuple_element_t<0, std::tuple<Operators...>>;

      public:

        static constexpr std::size_t output_field_size = FirstOperatorType::cfg_t::output_field_size;
        using field_t                                  = typename FirstOperatorType::field_t;

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
            bool is_first = true;
            for_each(m_operators,
                     [&](const auto& op)
                     {
                         ss << (is_first ? "" : " + ") << op.name();
                         is_first = false;
                     });
            return ss.str();
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
        // return std::apply(
        //     [&]
        //     {
        //         return OperatorSum<Operators...>(operator_tuple);
        //     });
        // return std::apply(OperatorSum<Operators...>, operator_tuple);
        return OperatorSum<Operators...>(operator_tuple);
    }

    template <class cfg, class bdry_cfg>
    FluxBasedScheme<cfg, bdry_cfg> operator+(const FluxBasedScheme<cfg, bdry_cfg>& scheme1, const FluxBasedScheme<cfg, bdry_cfg>& scheme2)
    {
        FluxBasedScheme<cfg, bdry_cfg> sum_scheme(scheme1); // copy
        sum_scheme.set_name(scheme1.name() + " + " + scheme2.name());

        static_for<0, cfg::dim>::apply(
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                if constexpr (cfg::scheme_type == SchemeType::LinearHomogeneous)
                {
                    sum_scheme.flux_definition()[d].flux_function = [=](auto h)
                    {
                        return scheme1.flux_definition()[d].flux_function(h) + scheme2.flux_definition()[d].flux_function(h);
                    };
                }
                else if constexpr (cfg::scheme_type == SchemeType::LinearHeterogeneous)
                {
                    sum_scheme.flux_definition()[d].flux_function = [=](auto& cells)
                    {
                        return scheme1.flux_definition()[d].flux_function(cells) + scheme2.flux_definition()[d].flux_function(cells);
                    };
                }
                else // SchemeType::NonLinear
                {
                    sum_scheme.flux_definition()[d].flux_function = [=](auto& cells, auto& field)
                    {
                        return scheme1.flux_definition()[d].flux_function(cells, field)
                             + scheme2.flux_definition()[d].flux_function(cells, field);
                    };
                }
            });
        return sum_scheme;
    }

    /**
     * Operator '+'
     */
    template <class cfg1,
              class bdry_cfg1,
              class cfg2,
              class bdry_cfg2,
              std::enable_if_t<!std::is_same_v<cfg1, cfg2> || !std::is_same_v<bdry_cfg1, bdry_cfg2>, bool> = true>
    auto operator+(const FluxBasedScheme<cfg1, bdry_cfg1>& scheme1, const FluxBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, scheme2);
    }

    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    auto operator+(const FluxBasedScheme<cfg1, bdry_cfg1>& scheme1, const CellBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, scheme2);
    }

    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    auto operator+(const CellBasedScheme<cfg1, bdry_cfg1>& scheme1, const FluxBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return make_operator_sum(scheme1, scheme2);
    }

    template <class Scheme, class... Operators>
    constexpr bool scheme_is_combinable = std::disjunction_v<std::is_same<Operators, Scheme>...>;

    template <class cfg, class bdry_cfg, class... Operators>
    auto operator+(const OperatorSum<Operators...>& sum_scheme, const CellBasedScheme<cfg, bdry_cfg>& scheme)
        -> std::enable_if_t<scheme_is_combinable<CellBasedScheme<cfg, bdry_cfg>, Operators...>, OperatorSum<Operators...>>
    {
        // static constexpr bool type_is_in_sum = std::disjunction_v<std::is_same<Operators, CellBasedScheme<cfg, bdry_cfg>>...>;

        for_each(sum_scheme.operators(),
                 [&](auto& op)
                 {
                     if constexpr (std::is_same_v<CellBasedScheme<cfg, bdry_cfg>, decltype(op)>)
                     {
                         op = op + scheme;
                     }
                 });
        return sum_scheme;
    }

    template <class cfg, class bdry_cfg, class... Operators>
    auto operator+(const OperatorSum<Operators...>& sum_scheme, const CellBasedScheme<cfg, bdry_cfg>& scheme)
        -> std::enable_if_t<!scheme_is_combinable<CellBasedScheme<cfg, bdry_cfg>, Operators...>,
                            OperatorSum<CellBasedScheme<cfg, bdry_cfg>, Operators...>>
    {
        auto added = std::tuple_cat(std::make_tuple(scheme), sum_scheme.operators());
        return make_operator_sum(added);
    }

    // template <class cfg, class bdry_cfg, class... Operators>
    // auto operator+(const OperatorSum<Operators...>& scheme1, const FluxBasedScheme<cfg, bdry_cfg>& scheme2)
    // {
    //     return scheme2 + scheme1;
    // }

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

    template <class cfg, class bdry_cfg, class... Operators>
    auto operator-(const OperatorSum<Operators...>& sum_scheme, const CellBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return sum_scheme + ((-1) * scheme);
    }

} // end namespace samurai
