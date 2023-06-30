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

    /**
     * constexpr power function
     */
    template <typename T>
    constexpr T ce_pow(T num, unsigned int pow)
    {
        return pow == 0 ? 1 : num * ce_pow(num, pow - 1);
    }

    /**
     * constexpr ceil function
     */
    template <typename float_type, typename std::enable_if<std::is_floating_point<float_type>::value>::type* = nullptr>
    constexpr int ce_ceil(float_type f)
    {
        const int i = static_cast<int>(f);
        return f > static_cast<float_type>(i) ? i + 1 : i;
    }

    /**
     * Iterates over the elements of a tuple
     */
    template <std::size_t Index = 0,                                                 // start iteration at 0 index
              typename TTuple,                                                       // the tuple type
              std::size_t Size = std::tuple_size_v<std::remove_reference_t<TTuple>>, // tuple size
              typename TCallable,                                                    // the callable to be invoked for each tuple item
              typename... TArgs                                                      // other arguments to be passed to the callable
              >
    void for_each(TTuple&& tuple, TCallable&& callable, TArgs&&... args)
    {
        if constexpr (Index < Size)
        {
            std::invoke(callable, args..., std::get<Index>(tuple));

            if constexpr (Index + 1 < Size)
            {
                for_each<Index + 1>(std::forward<TTuple>(tuple), std::forward<TCallable>(callable), std::forward<TArgs>(args)...);
            }
        }
    }

    /**
     * Transform tuple
     */
    template <typename... Ts, typename Func, size_t... Is>
    auto transform_impl(const std::tuple<Ts...>& t, Func&& f, std::index_sequence<Is...>)
    {
        return std::tuple<std::invoke_result_t<Func, Ts>...>{f(std::get<Is>(t))...};
    }

    template <typename... Ts, typename Func>
    auto transform(const std::tuple<Ts...>& t, Func&& f)
    {
        return transform_impl(t, std::forward<Func>(f), std::make_index_sequence<sizeof...(Ts)>{});
    }

    /**
     * Static for loop
     */
    template <int Begin, int End>
    struct static_for
    {
        template <typename Lambda>
        static inline constexpr void apply(Lambda&& f)
        {
            if (Begin < End)
            {
                f(std::integral_constant<int, Begin>{});
                static_for<Begin + 1, End>::apply(f);
            }
        }
    };

    template <int N>
    struct static_for<N, N>
    {
        template <typename Lambda>
        static inline constexpr void apply(Lambda&&)
        {
        }
    };
} // namespace samurai
