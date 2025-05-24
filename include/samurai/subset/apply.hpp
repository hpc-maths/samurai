// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <iostream>
#include <tuple>

#include "concepts.hpp"
#include "utils.hpp"

// namespace samurai::experimental
namespace samurai
{
    namespace detail
    {
        template <std::size_t dim, class Set, class Func, class Container>
        void apply_impl(Set&& global_set, Func&& func, Container& index)
        {
            auto set            = global_set.template get_local_set<dim>(global_set.level(), index);
            auto start_and_stop = global_set.template get_start_and_stop_function<dim>();

            if constexpr (dim != 1)
            {
                auto func_int = [&](const auto& interval)
                {
                    // std::cout << fmt::format("[apply_impl - dim {}] ", dim) << "apply_impl interval: " << interval << " global_set level:
                    // " << global_set.level() << std::endl;
                    for (auto i = interval.start; i < interval.end; ++i)
                    {
                        index[dim - 2] = i;
                        apply_impl<dim - 1>(std::forward<Set>(global_set), std::forward<Func>(func), index);
                    }
                };
                apply(set, start_and_stop, func_int);
            }
            else
            {
                auto func_int = [&](const auto& interval)
                {
                    func(interval, index);
                };
                apply(set, start_and_stop, func_int);
            }
        }
    }

    template <class Set, class Func>
    void apply(Set&& global_set, Func&& func)
    {
        constexpr std::size_t dim = std::decay_t<Set>::dim;
        xt::xtensor_fixed<int, xt::xshape<dim - 1>> index;
        // std::cout << fmt::format("[apply - dim {}] ", dim) << global_set.level() << std::endl;
        if (global_set.exist())
        {
            // std::cout << fmt::format("[apply - dim {}] ", dim) << "exist: " << std::endl;
            detail::apply_impl<dim>(std::forward<Set>(global_set), std::forward<Func>(func), index);
        }
    }

    template <class Set, class StartEnd, class Func>
        requires IsSetOp<Set> || IsIntervalListVisitor<Set>
    void apply(Set&& set, StartEnd&& start_and_stop, Func&& func)
    {
        using interval_t = typename std::decay_t<Set>::interval_t;
        using value_t    = typename interval_t::value_t;

        interval_t result;
        int r_ipos = 0;
        set.next(0, std::forward<StartEnd>(start_and_stop));
        auto scan = set.min();

        // std::cout << "[local apply] " << "scan: " << scan << std::endl;
        while (scan < sentinel<value_t> && !set.is_empty())
        {
            bool is_in = set.is_in(scan);

            // std::cout << "[local apply] " << "scan: " << scan << " is_in: " << is_in << std::endl;

            if (is_in && r_ipos == 0)
            {
                result.start = scan;
                r_ipos       = 1;
            }
            else if (!is_in && r_ipos == 1)
            {
                result.end = scan;
                r_ipos     = 0;

                // std::cout << "[local apply] " << "result: " << result << " ";
                auto true_result = set.shift() >= 0 ? result >> static_cast<std::size_t>(set.shift())
                                                    : result << -static_cast<std::size_t>(set.shift());
                // std::cout << "true_result: " << true_result << std::endl;
                func(true_result);
            }

            set.next(scan, std::forward<StartEnd>(start_and_stop));
            scan = set.min();
            // std::cout << "[local apply] " << "end scan: " << scan << std::endl;
        }
    }
}
