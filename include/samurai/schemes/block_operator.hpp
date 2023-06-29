#pragma once
#include "../utils.hpp"

namespace samurai
{
    template <std::size_t rows_, std::size_t cols_, class... Operators>
    class BlockOperator
    {
      public:

        static constexpr std::size_t rows = rows_;
        static constexpr std::size_t cols = cols_;

      protected:

        std::tuple<Operators...> m_operators;

      public:

        explicit BlockOperator(const Operators&... operators)
            : m_operators(operators...)
        {
            static constexpr std::size_t n_operators = sizeof...(operators);
            static_assert(n_operators == rows * cols, "The number of operators must correspond to rows*cols.");
        }

        auto& operators() const
        {
            return m_operators;
        }

        template <class Func>
        void for_each_operator(Func&& f)
        {
            std::size_t i = 0;
            for_each(m_operators,
                     [&](auto& op)
                     {
                         auto row = i / cols;
                         auto col = i % cols;
                         f(op, row, col);
                         i++;
                     });
        }

        template <class Func>
        void for_each_operator(Func&& f) const
        {
            std::size_t i = 0;
            for_each(m_operators,
                     [&](const auto& op)
                     {
                         auto row = i / cols;
                         auto col = i % cols;
                         f(op, row, col);
                         i++;
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
                    std::size_t i = 0;
                    for_each(unknown_tuple,
                             [&](auto& u)
                             {
                                 if (col == i)
                                 {
                                     if constexpr (!std::is_same_v<std::decay_t<decltype(u)>, typename std::decay_t<decltype(op)>::field_t>)
                                     {
                                         std::cerr << "unknown " << i << " (named '" << u.name() << "') is not compatible with the scheme ("
                                                   << row << ", " << col << ") (named '" << op.name() << "')" << std::endl;
                                         assert(false);
                                         exit(EXIT_FAILURE);
                                     }
                                 }
                                 i++;
                             });
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
    };

    template <std::size_t rows, std::size_t cols, class... Operators>
    auto make_block_operator(const Operators&... operators)
    {
        return BlockOperator<rows, cols, Operators...>(operators...);
    }

} // end namespace samurai
