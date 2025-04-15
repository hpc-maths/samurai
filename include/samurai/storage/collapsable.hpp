// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <type_traits>

namespace samurai
{
    //----------------------------------------------------------------//
    // CollapsableMatrix:                                             //
    // fixed-size matrix that reduces to a scalar if rows = cols = 1. //
    //----------------------------------------------------------------//
    namespace detail
    {

        template <class MatrixType, class value_type, std::size_t rows, std::size_t cols, bool can_collapse>
        struct CollapsableMatrix
        {
            using Type = MatrixType;
        };

        // Template specialization: if rows=cols=1, then just a scalar coefficient
        template <class MatrixType, class value_type>
        struct CollapsableMatrix<MatrixType, value_type, 1, 1, true>
        {
            using Type = value_type;
        };
    }

    template <class MatrixType, class value_type, std::size_t rows, std::size_t cols, bool can_collapse>
    using CollapsableMatrix = typename detail::CollapsableMatrix<MatrixType, value_type, rows, cols, can_collapse>::Type;

    //----------------------------------------------------------------//
    // CollapsableArray:                                              //
    // fixed-size matrix that reduces to a scalar if rows = cols = 1. //
    //----------------------------------------------------------------//
    namespace detail
    {
        template <class ArrayType, class value_type, std::size_t size, bool can_collapse>
        struct CollapsableArray
        {
            using Type = ArrayType;
        };

        // Template specialization: if size=1, then just a scalar coefficient
        template <class ArrayType, class value_type>
        struct CollapsableArray<ArrayType, value_type, 1, true>
        {
            using Type = value_type;
        };
    }

    template <class ArrayType, class value_type, std::size_t size, bool can_collapse>
    using CollapsableArray = typename detail::CollapsableArray<ArrayType, value_type, size, can_collapse>::Type;

    // Collapsable std::array
    template <class value_type, std::size_t size, bool can_collapse>
    using CollapsStdArray = typename detail::CollapsableArray<std::array<value_type, size>, value_type, size, can_collapse>::Type;

    template <class value_type>
    void fill(value_type& scalar, value_type value)
    {
        scalar = value;
    }

}
