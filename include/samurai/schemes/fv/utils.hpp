#pragma once
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
     * Matrix type
     */
    namespace detail
    {
        /**
         * Local matrix to store the coefficients of a vectorial field.
         */
        template <class value_type, std::size_t rows, std::size_t cols>
        struct LocalMatrix
        {
            using Type = xt::xtensor_fixed<value_type, xt::xshape<rows, cols>>;
        };

        /**
         * Template specialization: if rows=cols=1, then just a scalar coefficient
         */
        template <class value_type>
        struct LocalMatrix<value_type, 1, 1>
        {
            using Type = value_type;
        };

        /**
         * Fixed-size vector
         */
        template <class value_type, std::size_t size>
        struct FixedVector
        {
            using Type = xt::xtensor_fixed<value_type, xt::xshape<size>>;
        };

        /**
         * Template specialization: if size=1, then just a scalar coefficient
         */
        template <class value_type>
        struct FixedVector<value_type, 1>
        {
            using Type = value_type;
        };
    }

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
        using matrix_type = typename detail::LocalMatrix<value_type, rows, cols>::Type;
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
        using matrix_type = typename detail::LocalMatrix<value_type, rows, cols>::Type;
        return zeros<matrix_type>();
    }

} // end namespace samurai
