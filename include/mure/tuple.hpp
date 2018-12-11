#pragma once

#include <tuple>

#include <xtl/xmeta_utils.hpp>

template <std::size_t N, class T, class L>
struct repeat_as_tuple_impl
{
    using type_list = typename repeat_as_tuple_impl<N - 1, T, xtl::mpl::push_back_t<L, T>>::type_list;
};

template <class T, class L>
struct repeat_as_tuple_impl<0, T, L>
{
    using type_list = L;
};

template <std::size_t N, class T>
struct repeat_as_tuple
{
    using type_list = typename repeat_as_tuple_impl<N, T, xtl::mpl::vector<>>::type_list;
    using type = typename xtl::mpl::cast_t<type_list, std::tuple>;
};

template <std::size_t N, class T>
using repeat_as_tuple_t = typename repeat_as_tuple<N, T>::type;