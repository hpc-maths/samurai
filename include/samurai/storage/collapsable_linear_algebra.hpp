// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "containers_config.hpp"

namespace samurai
{

    // template <class matrix_type>
    // matrix_type zeros()
    // {
    //     matrix_type mat;
    //     mat.fill(0);
    //     return mat;
    // }

    // template <>
    // double zeros<double>()
    // {
    //     return 0;
    // }

    template <class matrix_type>
    matrix_type zeros()
    {
        if constexpr (std::is_floating_point_v<matrix_type>)
        {
            return 0;
        }
        else
        {
            matrix_type mat;
            mat.fill(0);
            return mat;
        }
    }

    template <class value_type, std::size_t rows, std::size_t cols, bool can_collapse>
    auto zeros()
    {
        using matrix_type = CollapsMatrix<value_type, rows, cols, can_collapse>;
        return zeros<matrix_type>();
    }

    template <class matrix_type>
    matrix_type eye()
    {
        // static constexpr auto s = typename matrix_type::shape_type();
        // return xt::eye(s[0]);
        if constexpr (std::is_floating_point_v<matrix_type>)
        {
            return 1;
        }
#ifdef SAMURAI_STATIC_MAT_CONTAINER_EIGEN3
        else if constexpr (is_eigen_matrix_v<matrix_type>)
        {
            return matrix_type::Identity();
        }
#endif
        else if constexpr (is_xtensor_matrix_v<matrix_type>)
        {
            static constexpr auto s = typename matrix_type::shape_type();
            return xt::eye(s[0]);
        }
        // static_assert(false, "eye() must be specified.");
    }

    // template <>
    // double eye<double>()
    // {
    //     return 1;
    // }

    // template <class value_type, std::size_t rows, std::size_t cols>
    // xtensor_static_matrix<value_type, rows, cols> eye<xtensor_static_matrix<value_type, rows, cols>>()
    // {
    //     static constexpr auto s = typename xtensor_static_matrix<value_type, rows, cols>::shape_type();
    //     return xt::eye(s[0]);
    // }

    template <class value_type, std::size_t rows, std::size_t cols, bool can_collapse>
    auto eye()
    {
        using matrix_type = CollapsMatrix<value_type, rows, cols, can_collapse>;
        return eye<matrix_type>();
        // matrix_type e;
        // if constexpr (rows == 1 && cols == 1)
        // {
        //     e = 1;
        // }
        // else
        // {
        //     for (std::size_t i = 0; i < rows; ++i)
        //     {
        //         e(i, i) = 1;
        //     }
        // }
        // return e;
    }

    template <bool SOA, class value_type, std::enable_if_t<std::is_floating_point_v<value_type>, bool> = true>
    auto mat_vec(value_type A, value_type x)
    {
        return A * x;
    }

    template <bool SOA, bool can_collapse, class value_type, std::size_t rows, std::size_t cols, class vector_type>
    auto mat_vec(const Matrix<value_type, rows, cols>& A, const vector_type& x)
    {
        // 'vector_type' can be an xt::view or a CollapsArray

        CollapsArray<value_type, rows, SOA, can_collapse> res = zeros<CollapsMatrix<value_type, rows, cols, can_collapse>>();
        if constexpr (rows == 1 && cols == 1 && can_collapse)
        {
            res = A * x;
        }
        else if constexpr (rows == 1)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                res += A(0, j) * x(j);
            }
        }
        else if constexpr (cols == 1)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                res(i) = A(i, 0) * x;
            }
        }
        else
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                for (std::size_t j = 0; j < cols; ++j)
                {
                    res(i) += A(i, j) * x(j);
                }
            }
        }
        return res;
    }

}
