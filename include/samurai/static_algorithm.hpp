// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace samurai
{
    template <std::size_t Dim, class TInterval>
    class LevelCellArray;

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

        template <size_t index_size, size_t dim, size_t dim_min>
        struct NestedLoop
        {
            using index_type = xt::xtensor_fixed<int, xt::xshape<index_size>>;

            template <typename Function>
            static constexpr void run(index_type& idx, int i0, int i1, Function&& func)
            {
                if constexpr (dim != dim_min - 1)
                {
                    for (idx[dim] = i0; idx[dim] != i1; ++idx[dim])
                    {
                        NestedLoop<index_size, dim - 1, dim_min>::run(idx, i0, i1, std::forward<Function>(func));
                    }
                }
                else
                {
                    func(idx);
                }
            }
        };

        template <size_t index_size, size_t dim, size_t dim_min>
        struct NestedExpand
        {
            static_assert(dim >= dim_min);
            using index_type = xt::xtensor_fixed<int, xt::xshape<index_size>>;

            template <class LCA_OR_SET>
            static auto run(index_type& idx, const LCA_OR_SET& lca, const int width)
            {
                if constexpr (dim != dim_min)
                {
                    idx[dim]       = -width;
                    auto subset_m1 = NestedExpand<index_size, dim - 1, dim_min>::run(idx, lca, width);
                    idx[dim]       = 0;
                    auto subset_0  = NestedExpand<index_size, dim - 1, dim_min>::run(idx, lca, width);
                    idx[dim]       = width;
                    auto subset_1  = NestedExpand<index_size, dim - 1, dim_min>::run(idx, lca, width);

                    return union_(subset_m1, subset_0, subset_1);
                }
                else
                {
                    idx[dim]       = -width;
                    auto subset_m1 = translate(lca, idx);
                    idx[dim]       = 0;
                    auto subset_0  = translate(lca, idx);
                    idx[dim]       = width;
                    auto subset_1  = translate(lca, idx);

                    return union_(subset_m1, subset_0, subset_1);
                }
            }
        };

    } // namespace detail

    template <class LCA_OR_SET, size_t dim_min = 0, size_t dim_max = LCA_OR_SET::dim>
    auto nestedExpand(const LCA_OR_SET& lca, const int width)
    {
        static constexpr std::size_t index_size = LCA_OR_SET::dim;
        using index_type                        = typename detail::NestedExpand<index_size, dim_max - 1, dim_min>::index_type;
        index_type idx;
        for (size_t i = 0; i != dim_min; ++i)
        {
            idx[i] = 0;
        }
        for (size_t i = dim_max; i != index_size; ++i)
        {
            idx[i] = 0;
        }
        return detail::NestedExpand<index_size, dim_max - 1, dim_min>::run(idx, lca, width);
    }

    template <size_t index_size, size_t dim_min, size_t dim_max, typename Function>
    inline void nestedLoop(int i0, int i1, Function&& func)
    {
        using index_type = typename detail::NestedLoop<index_size, dim_max - 1, dim_min>::index_type;
        index_type idx;
        for (size_t i = 0; i != dim_min; ++i)
        {
            idx[i] = i0;
        }
        for (size_t i = dim_max; i != index_size; ++i)
        {
            idx[i] = i0;
        }
        detail::NestedLoop<index_size, dim_max - 1, dim_min>::run(idx, i0, i1, std::forward<Function>(func));
    }

    template <size_t index_size, typename Function>
    inline void nestedLoop(int i0, int i1, Function&& func)
    {
        nestedLoop<index_size, 0, index_size>(i0, i1, std::forward<Function>(func));
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

    template <std::size_t nloops, class Func>
    inline void static_nested_loop(int start, int end, Func&& f)
    {
        static_nested_loop<nloops>(start, end, 1, std::forward<Func>(f));
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
