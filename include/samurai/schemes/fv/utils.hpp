#pragma once

#include "../../storage/std/algebraic_array.hpp"
#include <xtensor/xfixed.hpp>

namespace samurai
{
    enum class SchemeType
    {
        NonLinear,
        LinearHeterogeneous,
        LinearHomogeneous
    };

    /**
     * Actual data structures used
     */
    namespace data_structure
    {
        template <class value_type, std::size_t size>
#ifdef FLUX_CONTAINER_array
        using Array = AlgebraicArray<value_type, size>;
#else
        using Array = xt::xtensor_fixed<value_type, xt::xshape<size>>;
#endif

        template <class value_type, std::size_t rows, std::size_t cols>
        using Matrix = xt::xtensor_fixed<value_type, xt::xshape<rows, cols>>;
    }

    namespace detail
    {
        template <class value_type, std::size_t rows, std::size_t cols>
        struct FixedCollapsableMatrix
        {
            using Type = data_structure::Matrix<value_type, rows, cols>;
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
        struct FixedCollapsableArray
        {
            using Type = data_structure::Array<value_type, size>;
        };

        /**
         * Template specialization: if size=1, then just a scalar coefficient
         */
        template <class value_type>
        struct FixedCollapsableArray<value_type, 1>
        {
            using Type = value_type;
        };

        template <class value_type, std::size_t size>
        struct FixedCollapsableStdArray
        {
            using Type = std::array<value_type, size>;
        };

        /**
         * Template specialization: if size=1, then just a scalar coefficient
         */
        template <class value_type>
        struct FixedCollapsableStdArray<value_type, 1>
        {
            using Type = value_type;
        };
    }

    template <class value_type, std::size_t size>
    using Array = data_structure::Array<value_type, size>;

    template <class value_type, std::size_t rows, std::size_t cols>
    using Matrix = data_structure::Matrix<value_type, rows, cols>;

    /**
     * Collapsable, fixed-size matrix: reduces to a scalar if rows = cols = 1.
     */
    template <class value_type, std::size_t rows, std::size_t cols>
    using CollapsMatrix = typename detail::FixedCollapsableMatrix<value_type, rows, cols>::Type;
    /**
     * Collapsable, fixed size array: reduces to a scalar if size = 1.
     */
    template <class value_type, std::size_t size>
    using CollapsArray = typename detail::FixedCollapsableArray<value_type, size>::Type;

    /**
     * Collapsable, fixed size array: reduces to a scalar if size = 1.
     */
    template <class value_type, std::size_t size>
    using CollapsStdArray = typename detail::FixedCollapsableStdArray<value_type, size>::Type;

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
        // 'vector_type' can be an xt::view or a CollapsArray

        CollapsArray<value_type, rows> res = zeros<CollapsMatrix<value_type, rows, cols>>();
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

    template <class cfg>
    using StencilCells = CollapsStdArray<typename cfg::input_field_t::cell_t, cfg::stencil_size>;

    template <class cfg>
    using JacobianMatrix = CollapsMatrix<typename cfg::input_field_t::value_type, cfg::output_field_size, cfg::input_field_t::size>;

    template <class cfg>
    using StencilJacobian = Array<JacobianMatrix<cfg>, cfg::stencil_size>;

} // end namespace samurai
