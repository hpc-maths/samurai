// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../collapsable.hpp"
#include <xtensor/xfixed.hpp>

namespace samurai
{
    //------------------//
    // Type definitions //
    //------------------//

    template <class value_type, std::size_t size>
    using xtensor_static_array = xt::xtensor_fixed<value_type, xt::xshape<size>>;

    template <class value_type, std::size_t rows, std::size_t cols>
    using xtensor_static_matrix = xt::xtensor_fixed<value_type, xt::xshape<rows, cols>>;

    template <class value_type, std::size_t size>
    using xtensor_collapsable_static_array = CollapsableArray<xtensor_static_array<value_type, size>, value_type, size>;

    template <class value_type, std::size_t rows, std::size_t cols>
    using xtensor_collapsable_static_matrix = CollapsableMatrix<xtensor_static_matrix<value_type, rows, cols>, value_type, rows, cols>;

    // is_xtensor_matrix //
    template <typename>
    struct is_xtensor_matrix : std::false_type
    {
    };

    template <class value_type, std::size_t rows, std::size_t cols>
    struct is_xtensor_matrix<xtensor_static_matrix<value_type, rows, cols>> : std::true_type
    {
    };

    template <class T>
    inline constexpr auto is_xtensor_matrix_v = is_xtensor_matrix<T>::value;

    //-----------//
    // Functions //
    //-----------//

    template <class value_type, std::size_t size>
    void fill(xtensor_static_array<value_type, size>& array, value_type value)
    {
        array.fill(value);
    }

    // template <class value_type, std::size_t rows, std::size_t cols, class T>
    // auto row(const xtensor_static_matrix<value_type, rows, cols>& A, T i)
    // {
    //     return xt::row(A, i);
    // }

    template <class value_type, std::size_t rows, std::size_t cols, class T>
    inline auto row(xtensor_static_matrix<value_type, rows, cols>& A, T i)
    {
        return xt::row(A, static_cast<std::ptrdiff_t>(i));
    }

    // template <class value_type, std::size_t rows, std::size_t cols, class T>
    // auto col(const xtensor_static_matrix<value_type, rows, cols>& A, T i)
    // {
    //     return xt::col(A, i);
    // }

    template <class value_type, std::size_t rows, std::size_t cols, class T>
    inline auto col(xtensor_static_matrix<value_type, rows, cols>& A, T i)
    {
        return xt::col(A, static_cast<std::ptrdiff_t>(i));
    }

}
