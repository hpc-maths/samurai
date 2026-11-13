// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <utility>

#include <xtensor/containers/xfixed.hpp>
#include <xtensor/core/xexpression.hpp>

#include "../../samurai_config.hpp"
#include "../collapsable.hpp"
#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
#include "../eigen/eigen.hpp"
#endif

namespace samurai
{
    /**
     * Fixed-size array of N values with element-wise arithmetic.
     *
     * Every operation is eager and returns a new array by value. There is no
     * expression template, so a result never refers to its operands.
     * This is the type of the field values a flux function reads, and the
     * default flux container.
     *
     * It mixes with xtensor: an array is built from, assigned from and combined
     * with any 1D xtensor expression of N values, and it converts to
     * xt::xtensor_fixed. A mixed operation returns an array.
     */
    template <class T, std::size_t N>
    struct StdArrayWrapper
    {
        using value_type     = T;
        using size_type      = std::size_t;
        using iterator       = typename std::array<T, N>::iterator;
        using const_iterator = typename std::array<T, N>::const_iterator;
        using xtensor_type   = xt::xtensor_fixed<T, xt::xshape<N>>;

      private:

        std::array<T, N> _a;

        template <class E>
        void copy_from(const xt::xexpression<E>& e)
        {
            const auto& values = e.derived_cast();
            assert(values.dimension() == 1 && values.size() == N);
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] = static_cast<T>(values(i));
            }
        }

      public:

        StdArrayWrapper() = default;

        explicit StdArrayWrapper(const std::array<T, N>& a)
            : _a(a)
        {
        }

        explicit StdArrayWrapper(std::array<T, N>&& a)
            : _a(std::move(a))
        {
        }

        // The constraint keeps '{array}' a copy instead of a one-element list.
        template <class T2>
            requires std::is_convertible_v<T2, T>
        StdArrayWrapper(std::initializer_list<T2> list)
        {
            assert(list.size() == N);
            std::transform(list.begin(),
                           list.end(),
                           _a.begin(),
                           [](const T2& value)
                           {
                               return static_cast<T>(value);
                           });
        }

        // Any 1D xtensor expression of N values: container, view or function.
        template <class E>
            requires std::is_convertible_v<typename E::value_type, T>
        StdArrayWrapper(const xt::xexpression<E>& e) // cppcheck-suppress noExplicitConstructor
        {
            copy_from(e);
        }

#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
        template <class D>
        StdArrayWrapper(const Eigen::DenseBase<D>& e) // cppcheck-suppress noExplicitConstructor
        {
            assert(e.size() == static_cast<Eigen::Index>(N));
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] = static_cast<T>(e.derived()(static_cast<Eigen::Index>(i)));
            }
        }
#endif

        // Every component is set to 'value'.
        StdArrayWrapper(T value) // cppcheck-suppress noExplicitConstructor
        {
            _a.fill(value);
        }

        template <class E>
            requires std::is_convertible_v<typename E::value_type, T>
        StdArrayWrapper& operator=(const xt::xexpression<E>& e)
        {
            copy_from(e);
            return *this;
        }

        // For the code that names an xtensor type: 'xt::xtensor_fixed<...> x = array;'.
        operator xtensor_type() const
            requires std::is_arithmetic_v<T>
        {
            return to_xtensor();
        }

        // A copy as an xtensor container, to call an 'xt::' function on it.
        xtensor_type to_xtensor() const
            requires std::is_arithmetic_v<T>
        {
            xtensor_type x;
            std::copy(_a.begin(), _a.end(), x.begin());
            return x;
        }

        auto& array()
        {
            return _a;
        }

        const auto& array() const
        {
            return _a;
        }

        // Number of components, as in xtensor and Eigen containers.
        constexpr std::size_t size() const
        {
            return N;
        }

        iterator begin()
        {
            return _a.begin();
        }

        const_iterator begin() const
        {
            return _a.begin();
        }

        iterator end()
        {
            return _a.end();
        }

        const_iterator end() const
        {
            return _a.end();
        }

        T* data()
        {
            return _a.data();
        }

        const T* data() const
        {
            return _a.data();
        }

        T& operator()(std::size_t i)
        {
            return _a[i];
        }

        const T& operator()(std::size_t i) const
        {
            return _a[i];
        }

        T& operator[](std::size_t i)
        {
            return _a[i];
        }

        const T& operator[](std::size_t i) const
        {
            return _a[i];
        }

        void fill(T value)
        {
            _a.fill(value);
        }

        //----------------------//
        // Compound assignments //
        //----------------------//

        StdArrayWrapper& operator+=(const StdArrayWrapper& other)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] += other._a[i];
            }
            return *this;
        }

        StdArrayWrapper& operator-=(const StdArrayWrapper& other)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] -= other._a[i];
            }
            return *this;
        }

        StdArrayWrapper& operator*=(const StdArrayWrapper& other)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] *= other._a[i];
            }
            return *this;
        }

        StdArrayWrapper& operator/=(const StdArrayWrapper& other)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] /= other._a[i];
            }
            return *this;
        }

        template <class E>
        StdArrayWrapper& operator+=(const xt::xexpression<E>& e)
        {
            return *this += StdArrayWrapper(e);
        }

        template <class E>
        StdArrayWrapper& operator-=(const xt::xexpression<E>& e)
        {
            return *this -= StdArrayWrapper(e);
        }

        template <class E>
        StdArrayWrapper& operator*=(const xt::xexpression<E>& e)
        {
            return *this *= StdArrayWrapper(e);
        }

        template <class E>
        StdArrayWrapper& operator/=(const xt::xexpression<E>& e)
        {
            return *this /= StdArrayWrapper(e);
        }

        template <class Scalar>
            requires std::is_arithmetic_v<Scalar>
        StdArrayWrapper& operator+=(Scalar scalar)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] += scalar;
            }
            return *this;
        }

        template <class Scalar>
            requires std::is_arithmetic_v<Scalar>
        StdArrayWrapper& operator-=(Scalar scalar)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] -= scalar;
            }
            return *this;
        }

        template <class Scalar>
            requires std::is_arithmetic_v<Scalar>
        StdArrayWrapper& operator*=(Scalar scalar)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] *= scalar;
            }
            return *this;
        }

        template <class Scalar>
            requires std::is_arithmetic_v<Scalar>
        StdArrayWrapper& operator/=(Scalar scalar)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] /= scalar;
            }
            return *this;
        }

        //------------------------------------------------------------------//
        // Functions found by argument-dependent lookup: 'abs(a)', 'sum(a)' //
        //------------------------------------------------------------------//
        // They carry the names of their xtensor counterparts. Being friends,
        // they do not hide '::sqrt' and the like inside the samurai namespace.

        friend bool operator==(const StdArrayWrapper& a, const StdArrayWrapper& b)
        {
            return a._a == b._a;
        }

        friend std::ostream& operator<<(std::ostream& out, const StdArrayWrapper& a)
        {
            out << "{";
            for (std::size_t i = 0; i < N; ++i)
            {
                out << (i == 0 ? "" : ", ") << a._a[i];
            }
            return out << "}";
        }

        friend StdArrayWrapper abs(const StdArrayWrapper& a)
        {
            StdArrayWrapper b;
            for (std::size_t i = 0; i < N; ++i)
            {
                b._a[i] = std::abs(a._a[i]);
            }
            return b;
        }

        friend StdArrayWrapper sqrt(const StdArrayWrapper& a)
        {
            StdArrayWrapper b;
            for (std::size_t i = 0; i < N; ++i)
            {
                b._a[i] = std::sqrt(a._a[i]);
            }
            return b;
        }

        // Element-wise minimum. A scalar operand stands for a constant array.
        friend StdArrayWrapper minimum(const StdArrayWrapper& a, const StdArrayWrapper& b)
        {
            StdArrayWrapper c;
            for (std::size_t i = 0; i < N; ++i)
            {
                c._a[i] = std::min(a._a[i], b._a[i]);
            }
            return c;
        }

        // Element-wise maximum. A scalar operand stands for a constant array.
        friend StdArrayWrapper maximum(const StdArrayWrapper& a, const StdArrayWrapper& b)
        {
            StdArrayWrapper c;
            for (std::size_t i = 0; i < N; ++i)
            {
                c._a[i] = std::max(a._a[i], b._a[i]);
            }
            return c;
        }

        friend T sum(const StdArrayWrapper& a)
        {
            T s{};
            for (std::size_t i = 0; i < N; ++i)
            {
                s += a._a[i];
            }
            return s;
        }

        // Smallest component.
        friend T amin(const StdArrayWrapper& a)
        {
            return *std::min_element(a._a.begin(), a._a.end());
        }

        // Largest component.
        friend T amax(const StdArrayWrapper& a)
        {
            return *std::max_element(a._a.begin(), a._a.end());
        }

        // Euclidean norm.
        friend T norm_l2(const StdArrayWrapper& a)
        {
            T s{};
            for (std::size_t i = 0; i < N; ++i)
            {
                s += a._a[i] * a._a[i];
            }
            return std::sqrt(s);
        }
    };

    namespace detail
    {
        template <class>
        struct is_std_array_wrapper : std::false_type
        {
        };

        template <class T, std::size_t N>
        struct is_std_array_wrapper<StdArrayWrapper<T, N>> : std::true_type
        {
        };
    }

    // A StdArrayWrapper, whatever its constness and value category.
    template <class A>
    concept std_array_wrapper = detail::is_std_array_wrapper<std::decay_t<A>>::value;

    // An xtensor expression, whatever its constness and value category.
    template <class E>
    concept xtensor_expression = xt::is_xexpression<E>::value;

    //------------------//
    // array (op) array //
    //------------------//

    template <class T, std::size_t N>
    StdArrayWrapper<T, N> operator+(const StdArrayWrapper<T, N>& a, const StdArrayWrapper<T, N>& b)
    {
        StdArrayWrapper<T, N> c(a);
        c += b;
        return c;
    }

    template <class T, std::size_t N>
    StdArrayWrapper<T, N> operator-(const StdArrayWrapper<T, N>& a, const StdArrayWrapper<T, N>& b)
    {
        StdArrayWrapper<T, N> c(a);
        c -= b;
        return c;
    }

    template <class T, std::size_t N>
    StdArrayWrapper<T, N> operator*(const StdArrayWrapper<T, N>& a, const StdArrayWrapper<T, N>& b)
    {
        StdArrayWrapper<T, N> c(a);
        c *= b;
        return c;
    }

    template <class T, std::size_t N>
    StdArrayWrapper<T, N> operator/(const StdArrayWrapper<T, N>& a, const StdArrayWrapper<T, N>& b)
    {
        StdArrayWrapper<T, N> c(a);
        c /= b;
        return c;
    }

    template <class T, std::size_t N>
    StdArrayWrapper<T, N> operator-(const StdArrayWrapper<T, N>& a)
    {
        StdArrayWrapper<T, N> b;
        for (std::size_t i = 0; i < N; ++i)
        {
            b[i] = -a[i];
        }
        return b;
    }

    //-------------------//
    // array (op) scalar //
    //-------------------//
    // Each operator has its own loop. One written as a call to its mirror,
    // 'scalar * a' as 'a * scalar', is not always inlined by GCC, which
    // measurably slows a flux function down.

    template <class T, std::size_t N, class Scalar>
        requires std::is_arithmetic_v<Scalar>
    StdArrayWrapper<T, N> operator+(const StdArrayWrapper<T, N>& a, Scalar scalar)
    {
        StdArrayWrapper<T, N> b(a);
        b += scalar;
        return b;
    }

    template <class Scalar, class T, std::size_t N>
        requires std::is_arithmetic_v<Scalar>
    StdArrayWrapper<T, N> operator+(Scalar scalar, const StdArrayWrapper<T, N>& a)
    {
        StdArrayWrapper<T, N> b(a);
        b += scalar;
        return b;
    }

    template <class T, std::size_t N, class Scalar>
        requires std::is_arithmetic_v<Scalar>
    StdArrayWrapper<T, N> operator-(const StdArrayWrapper<T, N>& a, Scalar scalar)
    {
        StdArrayWrapper<T, N> b(a);
        b -= scalar;
        return b;
    }

    template <class Scalar, class T, std::size_t N>
        requires std::is_arithmetic_v<Scalar>
    StdArrayWrapper<T, N> operator-(Scalar scalar, const StdArrayWrapper<T, N>& a)
    {
        StdArrayWrapper<T, N> b;
        for (std::size_t i = 0; i < N; ++i)
        {
            b[i] = scalar - a[i];
        }
        return b;
    }

    template <class T, std::size_t N, class Scalar>
        requires std::is_arithmetic_v<Scalar>
    StdArrayWrapper<T, N> operator*(const StdArrayWrapper<T, N>& a, Scalar scalar)
    {
        StdArrayWrapper<T, N> b(a);
        b *= scalar;
        return b;
    }

    template <class Scalar, class T, std::size_t N>
        requires std::is_arithmetic_v<Scalar>
    StdArrayWrapper<T, N> operator*(Scalar scalar, const StdArrayWrapper<T, N>& a)
    {
        StdArrayWrapper<T, N> b(a);
        b *= scalar;
        return b;
    }

    template <class T, std::size_t N, class Scalar>
        requires std::is_arithmetic_v<Scalar>
    StdArrayWrapper<T, N> operator/(const StdArrayWrapper<T, N>& a, Scalar scalar)
    {
        StdArrayWrapper<T, N> b(a);
        b /= scalar;
        return b;
    }

    template <class Scalar, class T, std::size_t N>
        requires std::is_arithmetic_v<Scalar>
    StdArrayWrapper<T, N> operator/(Scalar scalar, const StdArrayWrapper<T, N>& a)
    {
        StdArrayWrapper<T, N> b;
        for (std::size_t i = 0; i < N; ++i)
        {
            b[i] = scalar / a[i];
        }
        return b;
    }

    //-------------------------------//
    // array (op) xtensor expression //
    //-------------------------------//
    // The expression is evaluated and the result is an array.
    //
    // xtensor declares 'operator+(E1&&, E2&&)' for any pair holding one xtensor
    // expression, and would take the array for a scalar. The operators below
    // have the same signature plus a constraint, which is what makes overload
    // resolution prefer them.

    template <class A, class E>
        requires std_array_wrapper<A> && xtensor_expression<E>
    auto operator+(A&& a, E&& e)
    {
        return a + std::decay_t<A>(e);
    }

    template <class E, class A>
        requires xtensor_expression<E> && std_array_wrapper<A>
    auto operator+(E&& e, A&& a)
    {
        return std::decay_t<A>(e) + a;
    }

    template <class A, class E>
        requires std_array_wrapper<A> && xtensor_expression<E>
    auto operator-(A&& a, E&& e)
    {
        return a - std::decay_t<A>(e);
    }

    template <class E, class A>
        requires xtensor_expression<E> && std_array_wrapper<A>
    auto operator-(E&& e, A&& a)
    {
        return std::decay_t<A>(e) - a;
    }

    template <class A, class E>
        requires std_array_wrapper<A> && xtensor_expression<E>
    auto operator*(A&& a, E&& e)
    {
        return a * std::decay_t<A>(e);
    }

    template <class E, class A>
        requires xtensor_expression<E> && std_array_wrapper<A>
    auto operator*(E&& e, A&& a)
    {
        return std::decay_t<A>(e) * a;
    }

    template <class A, class E>
        requires std_array_wrapper<A> && xtensor_expression<E>
    auto operator/(A&& a, E&& e)
    {
        return a / std::decay_t<A>(e);
    }

    template <class E, class A>
        requires xtensor_expression<E> && std_array_wrapper<A>
    auto operator/(E&& e, A&& a)
    {
        return std::decay_t<A>(e) / a;
    }

    //--------------------------------//
    // xtensor expression (op)= array //
    //--------------------------------//
    // 'x' is a 1D container or view of N values, temporary views included:
    // 'xt::col(matrix, j) -= array'. Without these operators, xtensor would
    // take the array for a scalar.

    template <class X, class T, std::size_t N>
        requires xtensor_expression<X>
    X& operator+=(X&& x, const StdArrayWrapper<T, N>& a)
    {
        assert(x.dimension() == 1 && x.size() == N);
        for (std::size_t i = 0; i < N; ++i)
        {
            x(i) += a[i];
        }
        return x;
    }

    template <class X, class T, std::size_t N>
        requires xtensor_expression<X>
    X& operator-=(X&& x, const StdArrayWrapper<T, N>& a)
    {
        assert(x.dimension() == 1 && x.size() == N);
        for (std::size_t i = 0; i < N; ++i)
        {
            x(i) -= a[i];
        }
        return x;
    }

    template <class X, class T, std::size_t N>
        requires xtensor_expression<X>
    X& operator*=(X&& x, const StdArrayWrapper<T, N>& a)
    {
        assert(x.dimension() == 1 && x.size() == N);
        for (std::size_t i = 0; i < N; ++i)
        {
            x(i) *= a[i];
        }
        return x;
    }

    template <class X, class T, std::size_t N>
        requires xtensor_expression<X>
    X& operator/=(X&& x, const StdArrayWrapper<T, N>& a)
    {
        assert(x.dimension() == 1 && x.size() == N);
        for (std::size_t i = 0; i < N; ++i)
        {
            x(i) /= a[i];
        }
        return x;
    }

    //-----------//
    // Functions //
    //-----------//

    template <class T, std::size_t N, class Scalar>
    StdArrayWrapper<T, N> pow(const StdArrayWrapper<T, N>& a, Scalar exponent)
    {
        StdArrayWrapper<T, N> b;
        for (std::size_t i = 0; i < N; ++i)
        {
            b[i] = std::pow(a[i], exponent);
        }
        return b;
    }

    template <class NumberType1,
              class NumberType2,
              typename std::enable_if_t<std::is_arithmetic_v<NumberType1> && std::is_arithmetic_v<NumberType2>, bool> = true>
    SAMURAI_INLINE auto pow(NumberType1 base, NumberType2 exponent)
    {
        return std::pow(base, exponent);
    }

    template <class T, std::size_t N>
    void fill(StdArrayWrapper<T, N>& array, T value)
    {
        array.fill(value);
    }

    template <class value_type, std::size_t size, bool can_collapse>
    using collapsable_algebraic_std_array = CollapsableArray<StdArrayWrapper<value_type, size>, value_type, size, can_collapse>;

} // end namespace samurai
