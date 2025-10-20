// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <xtensor/containers/xfixed.hpp>

#include "field_expression.hpp"
#include "utils.hpp"

namespace samurai
{
    template <template <std::size_t dim, class T> class OP, class... CT>
    class field_operator_function : public field_expression<field_operator_function<OP, CT...>>
    {
      public:

        static constexpr std::size_t dim = detail::compute_dim<CT...>();

        inline field_operator_function(CT&&... e)
            : m_e{std::forward<CT>(e)...}
        {
            m_scaling_factor = detail::extract_mesh(arguments()).scaling_factor();
        }

        template <class interval_t, class... index_t>
        inline auto operator()(std::size_t level, interval_t i, index_t... index) const
        {
            OP<dim, interval_t> op(level, i, index...);
            return apply(op);
        }

        template <class interval_t, class coord_index_t>
        inline auto operator()(std::size_t level, interval_t i, xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> index) const
        {
            OP<dim, interval_t> op(level, i, index);
            return apply(op);
        }

        const auto& arguments() const
        {
            return m_e;
        }

        const auto& scaling_factor() const
        {
            return m_scaling_factor;
        }

      private:

        template <class interval_t>
        inline auto apply(OP<dim, interval_t>& op) const
        {
            return apply_impl(std::make_index_sequence<sizeof...(CT)>(), op);
        }

        template <std::size_t... I, class interval_t>
        inline auto apply_impl(std::index_sequence<I...>, OP<dim, interval_t>& op) const
        {
            return op(std::integral_constant<std::size_t, dim>{}, std::get<I>(m_e)...);
        }

        std::tuple<CT...> m_e;
        double m_scaling_factor = 1;
    };

    template <template <std::size_t dim, class T> class OP, class... CT>
    inline auto make_field_operator_function(CT&&... e)
    {
        return field_operator_function<OP, CT...>(std::forward<CT>(e)...);
    }

    template <std::size_t dim, class TInterval>
    class field_operator_base
    {
      public:

        using interval_t    = TInterval;
        using coord_index_t = typename interval_t::coord_index_t;
        using array_index_t = xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>>;

        // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes,misc-non-private-member-variables-in-classes)
        std::size_t level = 0;
        interval_t i;
        coord_index_t j = 0, k = 0;
        array_index_t index;

        // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes,misc-non-private-member-variables-in-classes)

      protected:

        inline field_operator_base(std::size_t level_, const interval_t& interval, const array_index_t& index_)
            : level{level_}
            , i{interval}
            , index{index_}
        {
            if constexpr (dim > 1)
            {
                j = index_[0];
            }
            if constexpr (dim > 2)
            {
                k = index_[1];
            }
        }

        inline field_operator_base(std::size_t level_, const interval_t& interval)
            : level{level_}
            , i{interval}
            , index{}
        {
        }

        inline field_operator_base(std::size_t level_, const interval_t& interval, coord_index_t j_)
            : level{level_}
            , i{interval}
            , j{j_}
            , index{j_}
        {
        }

        inline field_operator_base(std::size_t level_, const interval_t& interval, coord_index_t j_, coord_index_t k_)
            : level{level_}
            , i{interval}
            , j{j_}
            , k{k_}
            , index{j_, k_}
        {
        }
    };

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define INIT_OPERATOR(NAME)                                                                  \
    using interval_t    = TInterval;                                                         \
    using coord_index_t = typename interval_t::coord_index_t;                                \
                                                                                             \
    using base          = ::samurai::field_operator_base<dim, interval_t>;                   \
    using array_index_t = typename base::array_index_t;                                      \
    using base::i;                                                                           \
    using base::j;                                                                           \
    using base::k;                                                                           \
    using base::level;                                                                       \
    using base::index;                                                                       \
                                                                                             \
    inline NAME(std::size_t level_, const interval_t& interval, const array_index_t& index_) \
        : base(level_, interval, index_)                                                     \
    {                                                                                        \
    }                                                                                        \
                                                                                             \
    template <class... index_t>                                                              \
    inline NAME(std::size_t level_, const interval_t& interval, const index_t&... index_)    \
        : base(level_, interval, index_...)                                                  \
    {                                                                                        \
    }
    // NOLINTEND(cppcoreguidelines-macro-usage)
} // namespace samurai
