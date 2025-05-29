// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "concepts.hpp"
#include "utils.hpp"

namespace samurai
{
    namespace detail
    {
        template <std::size_t dim, class Set, class Func, class Container>
        bool apply_impl(Set&& global_set, Func&& func, Container& index)
        {
            auto set            = global_set.template get_local_set<dim>(global_set.level(), index);
            auto start_and_stop = global_set.template get_start_and_stop_function<dim>();

            if constexpr (dim != 1)
            {
                auto func_int = [&](const auto& interval)
                {
                    for (auto i = interval.start; i < interval.end; ++i)
                    {
                        index[dim - 2] = i;
                        if (apply_impl<dim - 1>(std::forward<Set>(global_set), std::forward<Func>(func), index))
                        {
                            return true;
                        }
                    }
                    return false;
                };
                return apply(set, start_and_stop, func_int);
            }
            else
            {
                auto func_int = [&](const auto& interval)
                {
                    return func(interval, index);
                };
                return apply(set, start_and_stop, func_int);
            }
        }
    }

    template <class Set, class Func>
    void apply(Set&& global_set, Func&& user_func)
    {
        constexpr std::size_t dim = std::decay_t<Set>::dim;
        xt::xtensor_fixed<int, xt::xshape<dim - 1>> index;

        auto func = [&](const auto& interval, const auto& index)
        {
            user_func(interval, index);
            return false;
        };

        if (global_set.exist())
        {
            detail::apply_impl<dim>(std::forward<Set>(global_set), func, index);
        }
    }

    template <class Set>
    bool empty_check(Set&& global_set)
    {
        constexpr std::size_t dim = std::decay_t<Set>::dim;
        xt::xtensor_fixed<int, xt::xshape<dim - 1>> index;

        auto func = [](const auto&, const auto&)
        {
            return true;
        };

        if (global_set.exist())
        {
            return !detail::apply_impl<dim>(std::forward<Set>(global_set), func, index);
        }
        return true;
    }

    template <class Set, class StartEnd, class Func>
        requires IsSetOp<Set> || IsIntervalListVisitor<Set>
    bool apply(Set&& set, StartEnd&& start_and_stop, Func&& func)
    {
        using interval_t = typename std::decay_t<Set>::interval_t;
        using value_t    = typename interval_t::value_t;

        interval_t result;
        int r_ipos = 0;
        set.next(0, std::forward<StartEnd>(start_and_stop));
        auto scan = set.min();

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
                result.end = scan;
                r_ipos     = 0;

                auto true_result = set.shift() >= 0 ? result >> static_cast<std::size_t>(set.shift())
                                                    : result << -static_cast<std::size_t>(set.shift());
                if (func(true_result))
                {
                    return true;
                }
            }

            set.next(scan, std::forward<StartEnd>(start_and_stop));
            scan = set.min();
        }
        return false;
    }
}
