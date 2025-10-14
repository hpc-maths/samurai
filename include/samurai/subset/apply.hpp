// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"

namespace samurai
{
    namespace detail
    {
        template <class Set, class Func, class Index, std::size_t d>
        void apply_rec(const SetBase<Set>& set,
                       Func&& func,
                       Index& index,
                       std::integral_constant<std::size_t, d> d_ic,
                       typename SetBase<Set>::Workspace& workspace)
        {
            using traverser_t        = typename Set::template traverser_t<d>;
            using current_interval_t = typename traverser_t::current_interval_t;

            set.init_workspace(1, d_ic, workspace);

            for (traverser_t traverser = set.get_traverser(index, d_ic, workspace); !traverser.is_empty(); traverser.next_interval())
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
                        apply_rec(set, std::forward<Func>(func), index, std::integral_constant<std::size_t, d - 1>{}, workspace);
                    }
                }
            }
        }
    }

    template <class Set, class Func>
    void apply(const SetBase<Set>& set, Func&& func)
    {
        using Workspace = typename SetBase<Set>::Workspace;

        constexpr std::size_t dim = Set::dim;

        xt::xtensor_fixed<int, xt::xshape<dim - 1>> index;
        if (set.exist())
        {
            Workspace workspace;
            detail::apply_rec(set, std::forward<Func>(func), index, std::integral_constant<std::size_t, dim - 1>{}, workspace);
        }
    }
}
