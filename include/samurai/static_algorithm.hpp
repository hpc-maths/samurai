// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace samurai
{
    template <typename T, size_t size, T value>
    constexpr std::array<T, size> make_const_array()
    {
        std::array<T, size> temp{};
        for (int i = 0; i < size; ++i)
        {
            temp[i] = value;
        }
        return temp;
    }

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

        template <size_t nloops, size_t iloop, int start, int end, int step = 1>
        struct StaticNestedForLoop
        {
            static_assert(nloops >= iloop, "invalid nested for loop");

            template <typename Func, typename index_type>
            inline static void run(Func&& func, index_type& index)
            {
                if constexpr (nloops == iloop)
                {
                    func(index);
                }
                else
                {
                    for (index[iloop] = start; index[iloop] < end; ++index[iloop])
                    {
                        StaticNestedForLoop<nloops, iloop + 1, start, end, step>::run(std::forward<Func>(func), index);
                    }
                }
            }
        };

    } // namespace detail

    template <size_t nloops, int start, int end, int step, class Func>
    inline void staticNestedForLoop(Func&& func)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;
        // std::array<int, nloops> index;
        detail::StaticNestedForLoop<nloops, 0, start, end, step>::run(std::forward<Func>(func), index);
    }

    template <size_t nloops, int start, int end, class Func>
    inline void staticNestedForLoop(Func&& func)
    {
        xt::xtensor_fixed<int, xt::xshape<nloops>> index;
        // std::array<int, nloops> index;
        detail::StaticNestedForLoop<nloops, 0, start, end, 1>::run(std::forward<Func>(func), index);
    }

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

    template <std::size_t nloops, std::array<int, nloops> start, std::array<int, nloops> end, std::array<int, nloops> step, class Func>
    inline void static_nested_loop(Func&& f)
    {
        std::array<int, nloops> index{};

        detail::StaticNestedForLoop<nloops, start, end, step>::run(f, index);
    }

    template <std::size_t nloops, std::array<int, nloops> start, std::array<int, nloops> end, class Func>
    inline void static_nested_loop(Func&& f)
    {
        std::array<int, nloops> index{};

        detail::StaticNestedForLoop<nloops, start, end, make_const_array<int, nloops, 1>()>::run(f, index);
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
    template <std::size_t Index = 0, // start iteration at 0 index
              typename TCallable,    // the callable to be invoked for each tuple item
              typename... TupleTypes>
    void for_each(const std::tuple<TupleTypes...>& tuple, TCallable&& callable)
    {
        constexpr std::size_t Size = std::tuple_size_v<std::remove_reference_t<std::tuple<TupleTypes...>>>;
        if constexpr (Index < Size)
        {
            std::invoke(callable, std::get<Index>(tuple));

            if constexpr (Index + 1 < Size)
            {
                for_each<Index + 1>(tuple, std::forward<TCallable>(callable));
            }
        }
    }

    template <std::size_t Index = 0, // start iteration at 0 index
              typename TCallable,    // the callable to be invoked for each tuple item
              typename... TupleTypes>
    void for_each(std::tuple<TupleTypes...>& tuple, TCallable&& callable)
    {
        constexpr std::size_t Size = std::tuple_size_v<std::remove_reference_t<std::tuple<TupleTypes...>>>;
        if constexpr (Index < Size)
        {
            std::invoke(callable, std::get<Index>(tuple));

            if constexpr (Index + 1 < Size)
            {
                for_each<Index + 1>(tuple, std::forward<TCallable>(callable));
            }
        }
    }

    /**
     * Transform tuple
     */
    template <typename... Ts, typename Func, size_t... Is>
    auto transform_impl(const std::tuple<Ts...>& t, Func&& f, std::index_sequence<Is...>)
    {
        return std::make_tuple(f(std::get<Is>(t))...);
    }

    template <typename... Ts, typename Func>
    auto transform(const std::tuple<Ts...>& t, Func&& f)
    {
        return transform_impl(t, std::forward<Func>(f), std::make_index_sequence<sizeof...(Ts)>{});
    }

    template <typename... Ts, typename Func, size_t... Is>
    auto transform_impl(std::tuple<Ts...>& t, Func&& f, std::index_sequence<Is...>)
    {
        return std::make_tuple(f(std::get<Is>(t))...);
    }

    template <typename... Ts, typename Func>
    auto transform(std::tuple<Ts...>& t, Func&& f)
    {
        return transform_impl(t, std::forward<Func>(f), std::make_index_sequence<sizeof...(Ts)>{});
    }

    /**
     * Static for loop
     */
    template <std::size_t begin, std::size_t end>
    struct static_for
    {
        template <typename lambda_t, std::size_t... Is>
        static inline constexpr void apply_impl(lambda_t&& f, std::integer_sequence<std::size_t, Is...>)
        {
            (f(std::integral_constant<std::size_t, Is + begin>{}), ...);
        }

        template <typename lambda_t>
        static inline constexpr void apply([[maybe_unused]] lambda_t&& f)
        {
            if constexpr (begin <= end)
            {
                apply_impl(std::forward<lambda_t>(f), std::make_integer_sequence<std::size_t, end - begin>());
            }
        }
    };
} // namespace samurai
