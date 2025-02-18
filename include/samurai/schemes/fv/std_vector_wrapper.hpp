#pragma once
#include <initializer_list>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <utility> // for std::move
#include <vector>

namespace samurai
{

    template <class T>
    struct StdVectorWrapper
    {
        using value_type = T;

      private:

        std::vector<T> _a;

      public:

        StdVectorWrapper()
        {
        }

        explicit StdVectorWrapper(const std::vector<T>& a)
            : _a(a)
        {
        }

        explicit StdVectorWrapper(std::vector<T>&& a)
            : _a(std::move(a))
        {
        }

        template <class T2>
        StdVectorWrapper(std::initializer_list<T2> list)
            : _a(list)
        {
        }

        template <class... T2>
        StdVectorWrapper(xt::xfunction<T2...> xt) // cppcheck-suppress noExplicitConstructor
        {
            for (std::size_t i = 0; i < size(); ++i)
            {
                this->_a[i] = xt(i);
            }
        }

        template <class... T2>
        StdVectorWrapper(xt::xview<T2...> xt) // cppcheck-suppress noExplicitConstructor
        {
            for (std::size_t i = 0; i < size(); ++i)
            {
                this->_a[i] = xt(i);
            }
        }

        StdVectorWrapper(std::size_t size) // cppcheck-suppress noExplicitConstructor
            : _a(size)
        {
        }

        // StdVectorWrapper(T value) // cppcheck-suppress noExplicitConstructor
        // {
        //     _a.fill(value);
        // }

        auto& vector()
        {
            return _a;
        }

        void resize(std::size_t new_size, T value = T())
        {
            _a.resize(new_size, value);
        }

        void reserve(std::size_t new_capacity)
        {
            _a.reserve(new_capacity);
        }

        void clear()
        {
            _a.clear();
        }

        bool empty() const
        {
            return _a.empty();
        }

        std::size_t size() const
        {
            return _a.size();
        }

        void push_back(const T& value)
        {
            _a.push_back(value);
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

        auto operator=(T value)
        {
            fill(value);
        }

        template <class xt>
        auto operator=(const xt& other)
        {
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] = other(i);
            }
        }

        auto& operator+=(const StdVectorWrapper<T>& other)
        {
            if (size() != other.size())
            {
                throw std::invalid_argument("Vector sizes do not match for addition.");
            }
#pragma omp simd
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] += other._a[i];
            }
            return *this;
        }

        auto& operator*=(const StdVectorWrapper<T>& other)
        {
            if (size() != other.size())
            {
                throw std::invalid_argument("Vector sizes do not match for addition.");
            }
#pragma omp simd
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] *= other._a[i];
            }
            return *this;
        }

        auto& operator-=(const StdVectorWrapper<T>& other)
        {
            if (size() != other.size())
            {
                throw std::invalid_argument("Vector sizes do not match for addition.");
            }
#pragma omp simd
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] -= other._a[i];
            }
            return *this;
        }

        auto& operator/=(const StdVectorWrapper<T>& other)
        {
            if (size() != other.size())
            {
                throw std::invalid_argument("Vector sizes do not match for addition.");
            }
#pragma omp simd
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] /= other._a[i];
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator+=(NumberType scalar)
        {
#pragma omp simd
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] += scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator*=(NumberType scalar)
        {
#pragma omp simd
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] *= scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator-=(NumberType scalar)
        {
#pragma omp simd
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] -= scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator/=(NumberType scalar)
        {
#pragma omp simd
            for (std::size_t i = 0; i < size(); ++i)
            {
                _a[i] /= scalar;
            }
            return *this;
        }

        void fill(T value)
        {
            _a.fill(value);
        }
    };

    template <class NumberType, class T>
    auto operator*(NumberType scalar, const StdVectorWrapper<T>& a)
    {
        StdVectorWrapper<T> b(a);
        b *= scalar;
        return b;
    }

    template <class T, class NumberType>
    auto operator*(const StdVectorWrapper<T>& a, NumberType scalar)
    {
        return scalar * a;
    }

    template <class NumberType, class T>
    StdVectorWrapper<T>&& operator*(NumberType scalar, StdVectorWrapper<T>&& a)
    {
        a *= scalar;
        return std::move(a);
    }

    template <class T, class NumberType>
    StdVectorWrapper<T>&& operator*(StdVectorWrapper<T>&& a, NumberType scalar)
    {
        a *= scalar;
        return std::move(a);
    }

    template <class T, class NumberType>
    auto operator/(const StdVectorWrapper<T>& a, NumberType scalar)
    {
        StdVectorWrapper<T> b(a);
        b /= scalar;
        return b;
    }

    template <class T, class NumberType>
    auto operator/(NumberType scalar, const StdVectorWrapper<T>& a)
    {
        StdVectorWrapper<T> b(a.size());
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            b[i] = scalar / a[i];
        }
        return b;
    }

    template <class NumberType, class T>
    auto operator+(NumberType scalar, const StdVectorWrapper<T>& a)
    {
        StdVectorWrapper<T> b(a);
        b += scalar;
        return b;
    }

    template <class T, class NumberType>
    auto operator+(const StdVectorWrapper<T>& a, NumberType scalar)
    {
        return scalar + a;
    }

    template <class T>
    auto operator+(const StdVectorWrapper<T>& a, const StdVectorWrapper<T>& b)
    {
        StdVectorWrapper<T> c(a);
        c += b;
        return c;
    }

    template <class T>
    StdVectorWrapper<T>&& operator+(const StdVectorWrapper<T>& a, StdVectorWrapper<T>&& b)
    {
        b += a;
        return std::move(b);
    }

    template <class T>
    StdVectorWrapper<T>&& operator+(StdVectorWrapper<T>&& a, const StdVectorWrapper<T>& b)
    {
        a += b;
        return std::move(a);
    }

    template <class T>
    auto operator+(StdVectorWrapper<T>&& a, StdVectorWrapper<T>&& b)
    {
        a += b;
        return std::move(a);
    }

    template <class T>
    auto operator-(const StdVectorWrapper<T>& a, const StdVectorWrapper<T>& b)
    {
        StdVectorWrapper<T> c(a);
        c -= b;
        return c;
    }

    template <class T>
    StdVectorWrapper<T>&& operator-(StdVectorWrapper<T>&& a, const StdVectorWrapper<T>& b)
    {
        a -= b;
        return std::move(a);
    }

    template <class T>
    auto operator*(const StdVectorWrapper<T>& a, const StdVectorWrapper<T>& b)
    {
        StdVectorWrapper<T> c(a);
        c *= b;
        return c;
    }

    template <class T>
    auto operator/(const StdVectorWrapper<T>& a, const StdVectorWrapper<T>& b)
    {
        StdVectorWrapper<T> c(a);
        c /= b;
        return c;
    }

    template <class T>
    auto operator-(const StdVectorWrapper<T>& a)
    {
        return -1. * a;
    }

    template <class T, class NumberType>
    auto pow(const StdVectorWrapper<T>& a, NumberType exponent)
    {
        StdVectorWrapper<T> b(a.size());
#pragma omp simd
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            b[i] = std::pow(a[i], exponent);
        }
        return b;
    }

    template <class T, class NumberType>
    StdVectorWrapper<T>&& pow(StdVectorWrapper<T>&& a, NumberType exponent)
    {
#pragma omp simd
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            a[i] = std::pow(a[i], exponent);
        }
        return std::move(a);
    }

} // end namespace samurai
