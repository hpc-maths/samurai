// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "concepts.hpp"

#include "utils.hpp"

namespace samurai::experimental
{
    template <class Set, class Func>
    void apply(Set&& global_set, Func&& func)
    {
        auto set = global_set.get_local_set();
        apply(set, std::forward<Func>(func));
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
            // std::cout << std::boolalpha << "is_in: " << is_in << std::endl;

            if (is_in && r_ipos == 0)
            {
                result.start = scan;
                r_ipos       = 1;
            }
            else if (!is_in && r_ipos == 1)
            {
                result.end = scan;
                r_ipos     = 0;
                // std::cout << result << " " << set.shift() << std::endl;
                auto true_result = result >> static_cast<std::size_t>(set.shift());
                func(true_result);
            }

            set.next(scan);
            scan = set.min();
            // std::cout << "scan " << scan << std::endl;
        }
    }
}