#pragma once

#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace mure
{
    template<std::size_t nloops, int start, int end, int step,
             typename function_t>
    inline void
    static_nested_loop_impl(function_t &&f,
                            xt::xtensor_fixed<int, xt::xshape<nloops>> &index,
                            std::integral_constant<std::size_t, nloops>)
    {
        f(index);
    }

    template<std::size_t nloops, int start, int end, int step,
             typename function_t, std::size_t iloop>
    inline void
    static_nested_loop_impl(function_t &&f,
                            xt::xtensor_fixed<int, xt::xshape<nloops>> &index,
                            std::integral_constant<std::size_t, iloop>)
    {
        for (int i = start; i < end; i += step)
        {
            index[nloops - 1 - iloop] = i;
            static_nested_loop_impl<nloops, start, end, step>(
                std::forward<function_t>(f), index,
                std::integral_constant<std::size_t, iloop + 1>{});
        }
    }

    template<std::size_t nloops, int start, int end, int step = 1,
             typename function_t>
    inline void static_nested_loop(function_t &&f)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;
        static_nested_loop_impl<nloops, start, end, step>(
            std::forward<function_t>(f), index,
            std::integral_constant<std::size_t, 0>{});
    }

    template<std::size_t nloops, typename function_t>
    inline void
    static_nested_loop_impl(int start, int end, int step, function_t &&f,
                            xt::xtensor_fixed<int, xt::xshape<nloops>> &index,
                            std::integral_constant<std::size_t, nloops>)
    {
        f(index);
    }

    template<std::size_t nloops, typename function_t, std::size_t iloop>
    inline void
    static_nested_loop_impl(int start, int end, int step, function_t &&f,
                            xt::xtensor_fixed<int, xt::xshape<nloops>> &index,
                            std::integral_constant<std::size_t, iloop>)
    {
        for (int i = start; i < end; i += step)
        {
            index[nloops - 1 - iloop] = i;
            static_nested_loop_impl<nloops>(start, end, step,
                std::forward<function_t>(f), index,
                std::integral_constant<std::size_t, iloop + 1>{});
        }
    }

    template<std::size_t nloops, typename function_t>
    inline void static_nested_loop(int start, int end, int step, function_t &&f)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;
        static_nested_loop_impl<nloops>(start, end, step,
            std::forward<function_t>(f), index,
            std::integral_constant<std::size_t, 0>{});
    }
}