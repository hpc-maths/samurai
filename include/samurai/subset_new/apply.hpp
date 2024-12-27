// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>

#include "concepts.hpp"
#include "utils.hpp"

namespace samurai::experimental
{
    namespace detail
    {
        template <std::size_t dim, class Set, class Func, class Container>
        void apply_impl(Set&& global_set, Func&& func, Container& index)
        {
            auto set = global_set.template get_local_set<dim>(global_set.level(), index);

            if constexpr (dim != 1)
            {
                auto func_int = [&](auto& interval)
                {
                    for (auto i = interval.start; i < interval.end; ++i)
                    {
                        index[dim - 2] = i;
                        apply_impl<dim - 1>(std::forward<Set>(global_set), std::forward<Func>(func), index);
                    }
                };
                apply(set, func_int);
            }
            else
            {
                auto func_int = [&](auto& interval)
                {
                    func(interval, index);
                };
                apply(set, func_int);
            }
        }
    }

    template <class Set, class Func>
    void apply(Set&& global_set, Func&& func)
    {
        constexpr std::size_t dim = std::decay_t<Set>::dim;
        std::array<int, dim - 1> index;
        detail::apply_impl<dim>(std::forward<Set>(global_set), std::forward<Func>(func), index);
    }

    template <class Set, class Func>
        requires IsSetOp<Set> || IsIntervalVector<Set>
    void apply(Set&& set, Func&& func)
    {
        using interval_t = typename std::decay_t<Set>::interval_t;
        using value_t    = typename interval_t::value_t;

        interval_t result;
        int r_ipos = 0;
        set.next(0);
        auto scan = set.min();
        // std::cout << "first scan " << scan << std::endl;

        while (scan < sentinel<value_t> && !set.is_empty())
        {
            bool is_in = set.is_in(scan);

            if (is_in && r_ipos == 0)
            {
                result.start = scan;
                r_ipos       = 1;
            }
            else if (!is_in && r_ipos == 1)
            {
                result.end       = scan;
                r_ipos           = 0;
                auto true_result = set.shift() >= 0 ? result >> static_cast<std::size_t>(set.shift())
                                                    : result << -static_cast<std::size_t>(set.shift());
                // std::cout << result << " " << set.shift() << " " << true_result << std::endl;
                func(true_result);
            }

            set.next(scan);
            scan = set.min();
            // std::cout << "scan " << scan << std::endl;
        }
    }
}