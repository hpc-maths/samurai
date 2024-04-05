#pragma once
#include "algebraic_array.hpp"
#include <xtensor/xfixed.hpp>

namespace samurai
{
    enum class SchemeType
    {
        NonLinear,
        LinearHeterogeneous,
        LinearHomogeneous
    };

    template <class value_type, std::size_t size>
    // using Array = xt::xtensor_fixed<value_type, xt::xshape<size>>;
    using Array = AlgebraicArray<value_type, size>;

    /**
     * Matrix type
     */
    namespace detail
    {
        template <class value_type, std::size_t rows, std::size_t cols>
        struct FixedCollapsableMatrix
        {
            using Type = xt::xtensor_fixed<value_type, xt::xshape<rows, cols>>;
        };

        /**
         * Template specialization: if rows=cols=1, then just a scalar coefficient
         */
        template <class value_type>
        struct FixedCollapsableMatrix<value_type, 1, 1>
        {
            using Type = value_type;
        };

        template <class value_type, std::size_t size>
        struct FixedCollapsableVector
        {
            using Type = Array<value_type, size>;
        };

        /**
         * Template specialization: if size=1, then just a scalar coefficient
         */
        template <class value_type>
        struct FixedCollapsableVector<value_type, 1>
        {
            using Type = value_type;
        };

        template <class T, std::size_t size>
        struct FixedCollapsableArray
        {
            using Type = std::array<T, size>;
        };

        /**
         * Template specialization: if size=1, then just the object
         */
        template <class T>
        struct FixedCollapsableArray<T, 1>
        {
            using Type = T;
        };
    }

    /**
     * Collapsable, fixed-size matrix: reduces to a scalar if rows = cols = 1.
     */
    template <class value_type, std::size_t rows, std::size_t cols>
    using CollapsMatrix = typename detail::FixedCollapsableMatrix<value_type, rows, cols>::Type;

    /**
     * Collapsable, fixed size vector: reduces to a scalar if size = 1.
     */
    template <class value_type, std::size_t size>
    using CollapsVector = typename detail::FixedCollapsableVector<value_type, size>::Type;

    /**
     * Collapsable fixed size array: reduces to the object if size = 1.
     */
    template <class T, std::size_t size>
    using CollapsArray = typename detail::FixedCollapsableArray<T, size>::Type;

    template <class matrix_type>
    matrix_type eye()
    {
        static constexpr auto s = typename matrix_type::shape_type();
        return xt::eye(s[0]);
    }

    template <>
    double eye<double>()
    {
        return 1;
    }

    template <class value_type, std::size_t rows, std::size_t cols>
    auto eye()
    {
        using matrix_type = CollapsMatrix<value_type, rows, cols>;
        return eye<matrix_type>();
    }

    template <class matrix_type>
    matrix_type zeros()
    {
        matrix_type mat;
        mat.fill(0);
        return mat;
    }

    template <>
    double zeros<double>()
    {
        return 0;
    }

    template <class value_type, std::size_t rows, std::size_t cols>
    auto zeros()
    {
        using matrix_type = CollapsMatrix<value_type, rows, cols>;
        return zeros<matrix_type>();
    }

    template <class value_type, std::enable_if_t<std::is_floating_point_v<value_type>, bool> = true>
    auto mat_vec(value_type A, value_type x)
    {
        return A * x;
    }

    template <class value_type, std::size_t rows, std::size_t cols, class vector_type>
    auto mat_vec(const xt::xtensor_fixed<value_type, xt::xshape<rows, cols>>& A, const vector_type& x)
    {
        // 'vector_type' can be an xt::view or a CollapsVector

        CollapsVector<value_type, rows> res = zeros<CollapsMatrix<value_type, rows, cols>>();
        if constexpr (rows == 1 && cols == 1)
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

} // end namespace samurai
