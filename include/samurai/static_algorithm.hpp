// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace samurai
{
    // Static loop with boundaries known at compile time
    namespace detail
    {
        template <std::size_t nloops, int start, int end, int step, class Func>
        inline void
        static_nested_loop_impl(Func&& f, xt::xtensor_fixed<int, xt::xshape<nloops>>& index, std::integral_constant<std::size_t, nloops>)
        {
            f(index);
        }

        template <std::size_t nloops, int start, int end, int step, class Func, std::size_t iloop>
        inline void
        static_nested_loop_impl(Func&& f, xt::xtensor_fixed<int, xt::xshape<nloops>>& index, std::integral_constant<std::size_t, iloop>)
        {
            for (int i = start; i < end; i += step)
            {
                index[nloops - 1 - iloop] = i;

                static_nested_loop_impl<nloops, start, end, step>(std::forward<Func>(f),
                                                                  index,
                                                                  std::integral_constant<std::size_t, iloop + 1>{});
            }
        }
    } // namespace detail

    template <std::size_t nloops, int start, int end, int step, class Func>
    inline void static_nested_loop(Func&& f)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;

        detail::static_nested_loop_impl<nloops, start, end, step>(std::forward<Func>(f), index, std::integral_constant<std::size_t, 0>{});
    }

    template <std::size_t nloops, int start, int end, class Func>
    inline void static_nested_loop(Func&& f)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;

        detail::static_nested_loop_impl<nloops, start, end, 1>(std::forward<Func>(f), index, std::integral_constant<std::size_t, 0>{});
    }

    // Static loop with boundaries known at runtime
    namespace detail
    {
        template <std::size_t nloops, class Func>
        inline void static_nested_loop_impl(int /*start*/,
                                            int /*end*/,
                                            int /*step*/,
                                            Func&& f,
                                            xt::xtensor_fixed<int, xt::xshape<nloops>>& index,
                                            std::integral_constant<std::size_t, nloops>)
        {
            f(index);
        }

        template <std::size_t nloops, class Func, std::size_t iloop>
        inline void static_nested_loop_impl(int start,
                                            int end,
                                            int step,
                                            Func&& f,
                                            xt::xtensor_fixed<int, xt::xshape<nloops>>& index,
                                            std::integral_constant<std::size_t, iloop>)
        {
            for (int i = start; i < end; i += step)
            {
                index[nloops - 1 - iloop] = i;

                static_nested_loop_impl<nloops>(start, end, step, std::forward<Func>(f), index, std::integral_constant<std::size_t, iloop + 1>{});
            }
        }
    } // namespace detail

    template <std::size_t nloops, class Func>
    inline void static_nested_loop(int start, int end, int step, Func&& f)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;

        detail::static_nested_loop_impl<nloops>(start, end, step, std::forward<Func>(f), index, std::integral_constant<std::size_t, 0>{});
    }
} // namespace samurai