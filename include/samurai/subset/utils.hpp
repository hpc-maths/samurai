// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <functional>
#include <ranges>
#include <tuple>

#include <fmt/ranges.h>

#include "../list_of_intervals.hpp"

namespace samurai
{
    ////////////////////////////////////////////////////////////////////////
    //// misc
    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    const T& vmin(const T& a)
    {
        return a;
    }

    template <typename T>
    const T& vmax(const T& a)
    {
        return a;
    }

    template <typename T, typename... Ts>
    T vmin(const T& a, const T& b, const Ts&... others)
    {
        if constexpr (sizeof...(Ts) == 0u)
        {
            return std::min(a, b);
        }
        else
        {
            return a < b ? vmin(a, others...) : vmin(b, others...);
        }
    }

    template <typename T, typename... Ts>
    T vmax(const T& a, const T& b, const Ts&... others)
    {
        if constexpr (sizeof...(Ts) == 0u)
        {
            return std::max(a, b);
        }
        else
        {
            return a > b ? vmax(a, others...) : vmax(b, others...);
        }
    }

    namespace utils
    {

        template <typename T, std::size_t N>
        std::array<T, N> pow2(const std::array<T, N>& input, const std::size_t shift = 1)
        {
            std::array<T, N> output;
            for (std::size_t i = 0; i != N; ++i)
            {
                output[i] = input[i] << shift;
            }

            return output;
        }

        template <typename T, std::size_t N>
        std::array<T, N> sumAndPow2(const std::array<T, N>& input, const T& value, const std::size_t shift = 1)
        {
            std::array<T, N> output;
            for (std::size_t i = 0; i != N; ++i)
            {
                output[i] = (input[i] + value) << shift;
            }

            return output;
        }

        template <typename T, std::size_t N>
        std::array<T, N> powMinus2(const std::array<T, N>& input, const std::size_t shift = 1)
        {
            std::array<T, N> output;
            for (std::size_t i = 0; i != N; ++i)
            {
                output[i] = input[i] >> shift;
            }

            return output;
        }

        template <typename T, std::size_t N>
        xt::xtensor_fixed<T, xt::xshape<N>> pow2(const xt::xtensor_fixed<T, xt::xshape<N>>& input, const std::size_t shift = 1)
        {
            xt::xtensor_fixed<T, xt::xshape<N>> output;
            for (std::size_t i = 0; i != N; ++i)
            {
                output[i] = input[i] << shift;
            }

            return output;
        }

        template <typename T, std::size_t N>
        xt::xtensor_fixed<T, xt::xshape<N>>
        sumAndPow2(const xt::xtensor_fixed<T, xt::xshape<N>>& input, const T& value, const std::size_t shift = 1)
        {
            xt::xtensor_fixed<T, xt::xshape<N>> output;
            for (std::size_t i = 0; i != N; ++i)
            {
                output[i] = (input[i] + value) << shift;
            }

            return output;
        }

        template <typename T, std::size_t N>
        xt::xtensor_fixed<T, xt::xshape<N>> powMinus2(const xt::xtensor_fixed<T, xt::xshape<N>>& input, const std::size_t shift = 1)
        {
            xt::xtensor_fixed<T, xt::xshape<N>> output;
            for (std::size_t i = 0; i != N; ++i)
            {
                output[i] = input[i] >> shift;
            }

            return output;
        }

    }

    ////////////////////////////////////////////////////////////////////////
    //// intervals args
    ////////////////////////////////////////////////////////////////////////
    template <class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type::value_type>
    ForwardIt lower_bound_interval(ForwardIt begin, ForwardIt end, const T& value)
    {
        return std::lower_bound(begin,
                                end,
                                value,
                                [](const auto& interval, auto v)
                                {
                                    return interval.end <= v;
                                });
    }

    template <class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type::value_type>
    ForwardIt upper_bound_interval(ForwardIt begin, ForwardIt end, const T& value)
    {
        return std::upper_bound(begin,
                                end,
                                value,
                                [](auto v, const auto& interval)
                                {
                                    return v < interval.start;
                                });
    }

    ////////////////////////////////////////////////////////////////////////
    //// tuple iteration
    ////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <class Tuple, class Func, std::size_t... Is>
        Func&& enumerate_items(Tuple& tuple, Func&& func, std::index_sequence<Is...>)
        {
            (func(Is, std::get<Is>(tuple)), ...);
            return std::forward<Func>(func);
        }
    }

    template <class Tuple, class Func>
    Func&& enumerate_items(Tuple& tuple, Func&& func)
    {
        constexpr std::size_t N = std::tuple_size_v<std::decay_t<Tuple>>;

        return std::forward<Func>(detail::enumerate_items(tuple, func, std::make_index_sequence<N>{}));
    }

    template <std::ranges::forward_range Range, class Func>
    Func&& enumerate_items(Range& range, Func&& func)
    {
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202302L) || __cplusplus >= 202302L)
        for (auto& [i, item] : std::views::enumerate(range))
        {
            func(i, elem);
        }
#else
        size_t i = 0;
        for (auto& elem : range)
        {
            func(i, elem);
            ++i;
        }
#endif // if using c++23

        return std::forward<Func>(func);
    }

    ////////////////////////////////////////////////////////////////////////
    //// fill list of intervals
    ////////////////////////////////////////////////////////////////////////

    namespace subset_utils
    {
        namespace detail
        {
            template <typename Set, typename IndexRangeFunc, typename UnaryFunc, std::size_t D, std::size_t D_CUR>
            void transform_to_loi_rec(const Set& set,
                                      IndexRangeFunc&& indexRangeFunc,
                                      std::integral_constant<std::size_t, D> d,
                                      std::integral_constant<std::size_t, D_CUR> d_cur,
                                      UnaryFunc&& unaryFunc,
                                      typename Set::yz_index_t& index,
                                      typename Set::Workspace& child_workspace,
                                      ListOfIntervals<typename Set::value_t>& list_of_intervals)
            {
                using child_traverser_t        = typename Set::template traverser_t<d_cur>;
                using child_interval_t         = typename child_traverser_t::interval_t;
                using child_value_t            = typename child_traverser_t::value_t;
                using child_current_interval_t = typename child_traverser_t::current_interval_t;

                set.init_workspace(1, d_cur, child_workspace);

                for (child_traverser_t traverser = set.get_traverser(index, d_cur, child_workspace); !traverser.is_empty();
                     traverser.next_interval())
                {
                    const child_current_interval_t interval = traverser.current_interval();

                    if constexpr (d_cur == d)
                    {
                        list_of_intervals.add_interval(unaryFunc(d, interval));
                    }
                    else
                    {
                        const child_interval_t requested_interval = indexRangeFunc(d_cur);

                        // intersection between the current interval and the requested interval
                        const child_value_t index_start = std::max(interval.start, requested_interval.start);
                        const child_value_t index_bound = std::min(interval.end, requested_interval.end);

                        // recursive filling
                        for (index[d_cur - 1] = index_start; index[d_cur - 1] < index_bound; ++index[d_cur - 1])
                        {
                            transform_to_loi_rec(set,
                                                 std::forward<IndexRangeFunc>(indexRangeFunc),
                                                 d,
                                                 std::integral_constant<std::size_t, d_cur - 1>{},
                                                 std::forward<UnaryFunc>(unaryFunc),
                                                 index,
                                                 child_workspace,
                                                 list_of_intervals);
                        }
                    }
                }
            }
        } // namespace detail

        template <typename Set, typename IndexRangeFunc, typename UnaryFunc, std::size_t D>
        void transform_to_loi(const Set& set,
                              IndexRangeFunc&& indexRangeFunc,
                              std::integral_constant<std::size_t, D> d,
                              UnaryFunc&& unaryFunc,
                              typename Set::Workspace& child_workspace,
                              ListOfIntervals<typename Set::value_t>& list_of_intervals)
        {
            using yz_index_t = typename Set::yz_index_t;

            list_of_intervals.clear();

            yz_index_t index;
            index.fill(0); // to prevent -Wmaybe-uninitialized

            detail::transform_to_loi_rec(set,
                                         std::forward<IndexRangeFunc>(indexRangeFunc),
                                         d,
                                         std::integral_constant<std::size_t, Set::dim - 1>{},
                                         std::forward<UnaryFunc>(unaryFunc),
                                         index,
                                         child_workspace,
                                         list_of_intervals);
        }

    } // namespace subset_utils

} // namespace samurai
