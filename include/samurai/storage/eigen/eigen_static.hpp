// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../collapsable.hpp"
#include <Eigen/Core>

namespace samurai
{
    //------------------//
    // Type definitions //
    //------------------//

    namespace detail
    {
        template <class value_type, std::size_t size, bool SOA>
        struct eigen_static_array
        {
            using type = Eigen::Array<value_type, 1, size>;
        };

        template <class value_type, std::size_t size>
        struct eigen_static_array<value_type, size, true>
        {
            using type = Eigen::Array<value_type, size, 1>;
        };

        template <class value_type, std::size_t size, bool SOA>
        using eigen_static_array_t = typename eigen_static_array<value_type, size, SOA>::type;

    }
    template <class value_type, std::size_t size, bool SOA>
    using eigen_static_array = detail::eigen_static_array_t<value_type, size, SOA>;

    // using eigen_static_array = Eigen::Matrix<value_type, 1, size>;
    // using eigen_static_array = Eigen::Array<value_type, size, 1>;
    //  using eigen_static_array = Eigen::Matrix<value_type, size, 1>;
    //  using eigen_static_array = Eigen::Vector<value_type, size>;

    template <class value_type, std::size_t rows, std::size_t cols>
    using eigen_static_matrix = Eigen::Matrix<value_type, rows, cols>;

    template <class value_type, std::size_t size, bool SOA>
    using eigen_collapsable_static_array = CollapsableArray<eigen_static_array<value_type, size, SOA>, value_type, size>;

    template <class value_type, std::size_t rows, std::size_t cols>
    using eigen_collapsable_static_matrix = CollapsableMatrix<eigen_static_matrix<value_type, rows, cols>, value_type, rows, cols>;

    // is_eigen_matrix //
    template <typename>
    struct is_eigen_matrix : std::false_type
    {
    };

    template <class value_type, std::size_t rows, std::size_t cols>
    struct is_eigen_matrix<eigen_static_matrix<value_type, rows, cols>> : std::true_type
    {
    };

    template <class T>
    inline constexpr auto is_eigen_matrix_v = is_eigen_matrix<T>::value;

    //-----------//
    // Functions //
    //-----------//

    template <class value_type, std::size_t size, bool SOA>
    void fill(eigen_static_array<value_type, size, SOA>& array, value_type value)
    {
        array.fill(value);
    }

    template <class value_type, int rows, int cols>
    void fill(Eigen::Array<value_type, rows, cols>& array, value_type value)
    {
        array.fill(value);
    }

    template <class value_type, int rows, int cols, class T>
    auto row(const eigen_static_matrix<value_type, rows, cols>& A, T i)
    {
        return A.row(i);
    }

    template <class value_type, int rows, int cols, class T>
    auto row(eigen_static_matrix<value_type, rows, cols>& A, T i)
    {
        return A.row(i);
    }

    template <class value_type, int rows, int cols, class T>
    auto col(const eigen_static_matrix<value_type, rows, cols>& A, T i)
    {
        return A.col(i);
    }

    template <class value_type, int rows, int cols, class T>
    auto col(eigen_static_matrix<value_type, rows, cols>& A, T i)
    {
        return A.col(i);
    }

}
