// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"

namespace samurai
{
    namespace detail
    {
        template <class Set, class Func, std::size_t d>
        void apply_rec(const SetBase<Set>& set,
                       Func&& func,
                       typename SetBase<Set>::yz_index_t& yz_index,
                       std::integral_constant<std::size_t, d> d_ic,
                       typename SetBase<Set>::Workspace& workspace)
        {
            using traverser_t        = typename Set::template traverser_t<d>;
            using current_interval_t = typename traverser_t::current_interval_t;
            using interval_t         = typename traverser_t::interval_t;

            set.init_workspace(1, d_ic, workspace);

            //~for (traverser_t traverser = set.get_traverser(yz_index, d_ic, workspace); !traverser.is_empty(); traverser.next_interval())
            //~{
            //~    current_interval_t interval = traverser.current_interval();
            //~
            //~    if constexpr (d == 0)
            //~    {
            //~        func(interval, yz_index);
            //~    }
            //~    else
            //~    {
            //~        for (yz_index[d - 1] = interval.start; yz_index[d - 1] != interval.end; ++yz_index[d - 1])
            //~        {
            //~            apply_rec(set, std::forward<Func>(func), yz_index, std::integral_constant<std::size_t, d - 1>{}, workspace);
            //~        }
            //~    }
            //~}
            traverser_t traverser = set.get_traverser(yz_index, d_ic, workspace);
            if (!traverser.is_empty())
            {
                interval_t interval = traverser.current_interval();

                if constexpr (d == 0)
                {
                    func(interval, yz_index);
                }
                else
                {
                    for (yz_index[d - 1] = interval.start; yz_index[d - 1] != interval.end; ++yz_index[d - 1])
                    {
                        apply_rec(set, std::forward<Func>(func), yz_index, std::integral_constant<std::size_t, d - 1>{}, workspace);
                    }
                }
                traverser.next_interval();

                while (!traverser.is_empty())
                {
                    interval_t old_interval = interval;

                    interval = traverser.current_interval();

                    assert(old_interval < interval);

                    if constexpr (d == 0)
                    {
                        func(interval, yz_index);
                    }
                    else
                    {
                        for (yz_index[d - 1] = interval.start; yz_index[d - 1] != interval.end; ++yz_index[d - 1])
                        {
                            apply_rec(set, std::forward<Func>(func), yz_index, std::integral_constant<std::size_t, d - 1>{}, workspace);
                        }
                    }
                    traverser.next_interval();
                }
            }
        }
    }

    template <class Set, class Func>
    void apply(const SetBase<Set>& set, Func&& func)
    {
        using Workspace  = typename Set::Workspace;
        using yz_index_t = typename Set::yz_index_t;

        constexpr std::size_t dim = Set::dim;

        yz_index_t yz_index;
        if (set.exist())
        {
            Workspace workspace;
            detail::apply_rec(set, std::forward<Func>(func), yz_index, std::integral_constant<std::size_t, dim - 1>{}, workspace);
        }
    }
}
