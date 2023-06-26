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

        BlockOperator(const Operators&... operators)
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

        std::array<std::string, cols> field_names() const
        {
            std::array<std::string, cols> names;
            for_each_operator(
                [&](auto& op, auto row, auto col)
                {
                    if (row == col)
                    {
                        names[col] = op.unknown().name();
                    }
                });
            return names;
        }

        template <class... Fields>
        auto tie(Fields&... fields) const
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
