#pragma once

#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace mure
{
    // Static loop with boundaries known at compile time
    namespace detail
    {
        template<std::size_t nloops, int start, int end, int step, class Func>
        inline void static_nested_loop_impl(Func&& f,
                                xt::xtensor_fixed<int, xt::xshape<nloops>>& index,
                                std::integral_constant<std::size_t, nloops>)
        {
            f(index);
        }

        template<std::size_t nloops, int start, int end, int step, class Func, std::size_t iloop>
        inline void static_nested_loop_impl(Func&& f,
                                            xt::xtensor_fixed<int, xt::xshape<nloops>>& index,
                                            std::integral_constant<std::size_t, iloop>)
        {
            for (int i = start; i < end; i += step)
            {
                index[nloops - 1 - iloop] = i;

                static_nested_loop_impl<nloops, start, end, step>(
                    std::forward<Func>(f), index,
                    std::integral_constant<std::size_t, iloop + 1>{}
                );
            }
        }
    }

    template<std::size_t nloops, int start, int end, int step = 1, class Func>
    inline void static_nested_loop(Func&& f)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;

        detail::static_nested_loop_impl<nloops, start, end, step>(
            std::forward<Func>(f), index,
            std::integral_constant<std::size_t, 0>{}
        );
    }
    // Static loop with boundaries known at runtime
    namespace detail
    {
        template<std::size_t nloops, class Func>
        inline void static_nested_loop_impl(int start, int end, int step, Func&& f,
                                            xt::xtensor_fixed<int, xt::xshape<nloops>>& index,
                                            std::integral_constant<std::size_t, nloops>)
        {
            f(index);
        }

        template<std::size_t nloops, class Func, std::size_t iloop>
        inline void static_nested_loop_impl(int start, int end, int step, Func&& f,
                                            xt::xtensor_fixed<int, xt::xshape<nloops>> &index,
                                            std::integral_constant<std::size_t, iloop>)
        {
            for (int i = start; i < end; i += step)
            {
                index[nloops - 1 - iloop] = i;

                static_nested_loop_impl<nloops>(start, end, step,
                    std::forward<Func>(f), index,
                    std::integral_constant<std::size_t, iloop + 1>{}
                );
            }
        }
    }

    template<std::size_t nloops, class Func>
    inline void static_nested_loop(int start, int end, int step, Func&& f)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;

        detail::static_nested_loop_impl<nloops>(start, end, step,
            std::forward<Func>(f), index,
            std::integral_constant<std::size_t, 0>{}
        );
    }
}