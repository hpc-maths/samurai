#pragma once
#include "../utils.hpp"

namespace samurai
{
    // Concept to check if a type has cfg_t::scheme_type
    template <typename T>
    concept HasSchemeType = requires {
        typename T::cfg_t;
        T::cfg_t::scheme_type;
    };

    template <typename T>
    concept HasOutputFieldType = requires { typename T::output_field_t; };

    // Concept to check if a type is a BlockOperator
    template <typename T>
    concept IsBlockOperator = requires {
        T::rows;
        T::cols;
        std::declval<T>().operators();
    } && requires(T t) {
        { T::rows } -> std::convertible_to<std::size_t>;
        { T::cols } -> std::convertible_to<std::size_t>;
    };

    // Helper to get scheme type or return a default for non-operator types
    template <typename T>
    constexpr SchemeType get_scheme_type()
    {
        if constexpr (HasSchemeType<T>)
        {
            return T::cfg_t::scheme_type;
        }
        else
        {
            // For `int` (zero block) or other types without scheme_type,
            // treat as LinearHomogeneous
            return SchemeType::LinearHomogeneous;
        }
    }

    // Helper to extract diagonal operators' output field types
    template <std::size_t rows, std::size_t cols, typename... Operators>
    struct diagonal_output_fields
    {
      private:

        static constexpr std::size_t min_dim = std::min(rows, cols);

        // Get the type of the operator at position (i,i) in the operator tuple
        template <std::size_t i>
        using diagonal_op_type = std::tuple_element_t<i * cols + i, std::tuple<Operators...>>;

        // Helper to safely get output field type
        template <typename T>
        struct get_output_field_type
        {
            using type = void; // Default for types without output_field_t
        };

        template <typename T>
            requires HasOutputFieldType<T>
        struct get_output_field_type<T>
        {
            using type = typename T::output_field_t;
        };

        // Helper to build tuple type from diagonal field types recursively
        template <std::size_t I, std::size_t Max>
        struct build_diagonal_tuple
        {
            using current_field = std::conditional_t < I<Max && HasOutputFieldType<diagonal_op_type<I>>,
                                                         std::tuple<typename get_output_field_type<diagonal_op_type<I>>::type&>,
                                                         std::tuple<>>;
            using rest          = typename build_diagonal_tuple<I + 1, Max>::type;
            using type          = decltype(std::tuple_cat(std::declval<current_field>(), std::declval<rest>()));
        };

        // Specialization for the end case
        template <std::size_t Max>
        struct build_diagonal_tuple<Max, Max>
        {
            using type = std::tuple<>;
        };

      public:

        using type = typename build_diagonal_tuple<0, min_dim>::type;
    };

    template <std::size_t rows, std::size_t cols, typename... Operators>
    using diagonal_output_fields_t = typename diagonal_output_fields<rows, cols, Operators...>::type;

    // Helper to remove references from tuple elements
    template <typename T>
    struct remove_tuple_references;

    template <typename... Args>
    struct remove_tuple_references<std::tuple<Args...>>
    {
        using type = std::tuple<std::remove_reference_t<Args>...>;
    };

    template <class... Operators>
    constexpr SchemeType scheme_type_of_block_operator()
    {
        constexpr bool
            one_is_nonlinear = std::disjunction_v<std::integral_constant<bool, get_scheme_type<Operators>() == SchemeType::NonLinear>...>;
        constexpr bool one_is_linear_het = std::disjunction_v<
            std::integral_constant<bool, get_scheme_type<Operators>() == SchemeType::LinearHeterogeneous>...>;

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

    template <std::size_t rows_, std::size_t cols_, class... Operators>
    class BlockOperator
    {
        //   private:

        //     template <SchemeType scheme_type_>
        //     struct Config
        //     {
        //         static constexpr SchemeType scheme_type = scheme_type_;
        //     };

      public:

        static constexpr std::size_t rows = rows_;
        static constexpr std::size_t cols = cols_;

        // using cfg_t = Config<scheme_type_of_block_operator<Operators...>()>;

        using output_field_t = diagonal_output_fields_t<rows_, cols_, Operators...>;
        using input_field_t  = output_field_t; // simplification: in all cases so far, input and output fields are the same

        using output_field_no_ref_t = typename remove_tuple_references<output_field_t>::type;

      protected:

        std::tuple<Operators...> m_operators;

      public:

        explicit BlockOperator(const Operators&... operators)
            : m_operators(operators...)
        {
            static constexpr std::size_t n_operators = sizeof...(operators);
            static_assert(n_operators == rows * cols, "The number of operators must correspond to rows*cols.");
        }

        const auto& operators() const
        {
            return m_operators;
        }

        auto& operators()
        {
            return m_operators;
        }

        template <std::size_t row, std::size_t col>
        auto& get()
        {
            return std::get<row * cols + col>(m_operators);
        }

        template <std::size_t row, std::size_t col>
        const auto& get() const
        {
            return std::get<row * cols + col>(m_operators);
        }

        template <class Func>
        void for_each_operator(Func&& f)
        {
            static_for<0, rows>::apply(
                [&](auto row)
                {
                    static_for<0, cols>::apply(
                        [&](auto col)
                        {
                            f(get<row, col>(), row, col);
                        });
                });
        }

        template <class Func>
        void for_each_operator(Func&& f) const
        {
            static_for<0, rows>::apply(
                [&](auto row)
                {
                    static_for<0, cols>::apply(
                        [&](auto col)
                        {
                            f(get<row, col>(), row, col);
                        });
                });
        }

      public:

        template <class... Fields>
        auto tie_unknowns(Fields&... fields) const
        {
            static constexpr std::size_t n_fields = sizeof...(fields);
            static_assert(n_fields == cols, "The number of fields must correspond to the number of columns of the block operator.");

            auto unknown_tuple = std::tuple<Fields&...>(fields...);
            for_each_operator(
                [&](auto& op, auto row, auto col)
                {
                    using operator_t = std::decay_t<decltype(op)>;

                    auto& u = std::get<col>(unknown_tuple);

                    // Verify type compatibility only if not ZeroBlock
                    if constexpr (!std::is_same_v<operator_t, int>)
                    {
                        using op_field_t = typename operator_t::input_field_t;
                        if constexpr (!std::is_same_v<std::decay_t<decltype(u)>, op_field_t>)
                        {
                            std::cerr << "unknown " << col << " is not compatible with the scheme (" << row << ", " << col << ") (named '"
                                      << op.name() << "')" << std::endl;
                            assert(false);
                            exit(EXIT_FAILURE);
                        }
                    }
                });
            return unknown_tuple;
        }

        template <class... Fields>
        auto tie_rhs(Fields&... fields) const
        {
            static constexpr std::size_t n_fields = sizeof...(fields);
            static_assert(n_fields == rows, "The number of fields must correspond to the number of rows of the block operator.");

            return std::tuple<Fields&...>(fields...);
        }

        template <class... OutputFields, class... InputFields>
        void apply(std::tuple<OutputFields&...>& output_field, std::tuple<InputFields&...>& input_field)
        {
            static_assert(sizeof...(OutputFields) == rows,
                          "The number of output fields must correspond to the number of rows of the block operator.");
            static_assert(sizeof...(InputFields) == cols,
                          "The number of input fields must correspond to the number of columns of the block operator.");

            static_for<0, rows>::apply(
                [&](auto row)
                {
                    static_for<0, cols>::apply(
                        [&](auto col)
                        {
                            auto& op = get<row, col>();
                            op.apply_to(std::get<row>(output_field), std::get<col>(input_field));
                        });
                });
        }

        template <class... OutputFields, class... InputFields>
        void apply(std::tuple<OutputFields...>& output_field, std::tuple<InputFields&...>& input_field)
        {
            static_assert(sizeof...(OutputFields) == rows,
                          "The number of output fields must correspond to the number of rows of the block operator.");
            static_assert(sizeof...(InputFields) == cols,
                          "The number of input fields must correspond to the number of columns of the block operator.");

            static_for<0, rows>::apply(
                [&](auto row)
                {
                    static_for<0, cols>::apply(
                        [&](auto col)
                        {
                            auto& op = get<row, col>();
                            if constexpr (!std::is_same_v<std::decay_t<decltype(op)>, int>) // skip zero blocks
                            {
                                // std::cout << "Applying operator at (" << row << ", " << col << ") " << op.name() << std::endl;
                                op.apply(std::get<row>(output_field), std::get<col>(input_field));
                            }
                        });
                });
        }
    };

    template <std::size_t rows, std::size_t cols, class... Operators>
    auto make_block_operator(const Operators&... operators)
    {
        return BlockOperator<rows, cols, Operators...>(operators...);
    }

} // end namespace samurai
