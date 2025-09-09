// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"

namespace samurai
{
    namespace detail
    {
        template <Set_concept Set, class Func, class Index, std::size_t d>
        void apply_rec(const Set& set, Func&& func, Index& index, std::integral_constant<std::size_t, d> d_ic)
        {
            using traverser_t        = typename Set::template traverser_t<d>;
            using current_interval_t = typename traverser_t::current_interval_t;

            for (traverser_t traverser = set.get_traverser(index, d_ic); !traverser.is_empty(); traverser.next_interval())
            {
                current_interval_t interval = traverser.current_interval();
                if constexpr (d == 0)
                {
                    func(interval, index);
                }
                else
                {
                    for (index[d - 1] = interval.start; index[d - 1] != interval.end; ++index[d - 1])
                    {
                        apply_rec(set, std::forward<Func>(func), index, std::integral_constant<std::size_t, d - 1>{});
                    }
                }
            }
        }
    }

    template <Set_concept Set, class Func>
    void apply(const Set& set, Func&& func)
    {
        constexpr std::size_t dim = SetTraits<Set>::getDim();

        xt::xtensor_fixed<int, xt::xshape<dim - 1>> index;
        if (set.exist())
        {
            detail::apply_rec(set, std::forward<Func>(func), index, std::integral_constant<std::size_t, dim - 1>{});
        }
    }
}
