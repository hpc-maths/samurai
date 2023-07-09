// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xtensor/xfixed.hpp>

#include "field_expression.hpp"
#include "utils.hpp"

namespace samurai
{
    template <template <class T> class OP, class... CT>
    class field_operator_function : public field_expression<field_operator_function<OP, CT...>>
    {
      public:

        static constexpr std::size_t dim = detail::compute_dim<CT...>();

        inline field_operator_function(CT&&... e)
            : m_e{std::forward<CT>(e)...}
        {
        }

        template <class interval_t, class... index_t>
        inline auto operator()(std::size_t level, interval_t i, index_t... index) const
        {
            OP<interval_t> op(level, i, index...);
            return apply(op);
        }

        template <class interval_t, class coord_index_t>
        inline auto operator()(std::size_t level, interval_t i, xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index) const
        {
            OP<interval_t> op(level, i, index);
            return apply(op);
        }

        const auto& arguments() const
        {
            return m_e;
        }

      private:

        template <class interval_t>
        inline auto apply(OP<interval_t>& op) const
        {
            return apply_impl(std::make_index_sequence<sizeof...(CT)>(), op);
        }

        template <std::size_t... I, class interval_t>
        inline auto apply_impl(std::index_sequence<I...>, OP<interval_t>& op) const
        {
            return op(std::integral_constant<std::size_t, dim>{}, std::get<I>(m_e)...);
        }

        std::tuple<CT...> m_e;
    };

    template <template <class T> class OP, class... CT>
    inline auto make_field_operator_function(CT&&... e)
    {
        return field_operator_function<OP, CT...>(std::forward<CT>(e)...);
    }

    template <class TInterval>
    class field_operator_base
    {
      public:

        using interval_t    = TInterval;
        using coord_index_t = typename interval_t::coord_index_t;

        // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes,misc-non-private-member-variables-in-classes)
        std::size_t level = 0;
        interval_t i;
        coord_index_t j = 0, k = 0;

        // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes,misc-non-private-member-variables-in-classes)

        double dx() const
        {
            return m_dx;
        }

      protected:

        template <std::size_t dim>
        inline field_operator_base(std::size_t level_, const interval_t& interval, xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index)
            : level{level_}
            , i{interval}
            , m_dx{cell_length(level)}
        {
            if constexpr (dim > 0)
            {
                j = index[0];
            }
            if constexpr (dim > 1)
            {
                k = index[1];
            }
        }

        inline field_operator_base(std::size_t level_, const interval_t& interval)
            : level{level_}
            , i{interval}
            , m_dx{cell_length(level)}
        {
        }

        inline field_operator_base(std::size_t level_, const interval_t& interval, coord_index_t j_)
            : level{level_}
            , i{interval}
            , j{j_}
            , m_dx{cell_length(level)}
        {
        }

        inline field_operator_base(std::size_t level_, const interval_t& interval, coord_index_t j_, coord_index_t k_)
            : level{level_}
            , i{interval}
            , j{j_}
            , k{k_}
            , m_dx{cell_length(level)}
        {
        }

      private:

        double m_dx;
    };

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define INIT_OPERATOR(NAME)                                                                                                     \
    using interval_t    = TInterval;                                                                                            \
    using coord_index_t = typename interval_t::coord_index_t;                                                                   \
                                                                                                                                \
    using base = ::samurai::field_operator_base<interval_t>;                                                                    \
    using base::i;                                                                                                              \
    using base::j;                                                                                                              \
    using base::k;                                                                                                              \
    using base::level;                                                                                                          \
    using base::dx;                                                                                                             \
                                                                                                                                \
    template <std::size_t dim>                                                                                                  \
    inline NAME(std::size_t level_, const interval_t& interval, const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& index) \
        : base(level_, interval, index)                                                                                         \
    {                                                                                                                           \
    }                                                                                                                           \
                                                                                                                                \
    template <class... index_t>                                                                                                 \
    inline NAME(std::size_t level_, const interval_t& interval, const index_t&... index)                                        \
        : base(level_, interval, index...)                                                                                      \
    {                                                                                                                           \
    }
    // NOLINTEND(cppcoreguidelines-macro-usage)
} // namespace samurai
