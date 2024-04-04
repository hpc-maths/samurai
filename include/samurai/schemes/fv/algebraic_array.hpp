#pragma once
#include <array>
#include <math.h>

namespace samurai
{
    /**
     * Simple array with algebraic operations
     */
    template <class T, std::size_t size>
    struct AlgebraicArray
    {
        using value_type = T;

      private:

        std::array<T, size> _a;

      public:

        AlgebraicArray()
        {
        }

        explicit AlgebraicArray(const std::array<T, size>& a)
            : _a(a)
        {
        }

        explicit AlgebraicArray(std::array<T, size>&& a)
            : _a(std::move(a))
        {
        }

        template <class T2>
        AlgebraicArray(std::initializer_list<T2> list)
        {
            assert(list.size() == size);
            auto it = list.begin();
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] = *it;
                it++;
            }
        }

        template <class... T2>
        AlgebraicArray(xt::xfunction<T2...> xt)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                this->_a[i] = xt(i);
            }
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

        template <class xt>
        auto operator=(const xt& other)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] = other(i);
            }
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator*=(NumberType s)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] *= s;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator+=(NumberType s)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] += s;
            }
            return *this;
        }
    };

    template <class NumberType, class T, std::size_t size>
    auto operator*(NumberType scalar, const AlgebraicArray<T, size>& a)
    {
        AlgebraicArray<T, size> a2;
        for (std::size_t i = 0; i < size; ++i)
        {
            a2[i] = scalar * a[i];
        }
        return a2;
    }

    template <class T, std::size_t size, class NumberType>
    auto operator*(const AlgebraicArray<T, size>& a, NumberType scalar)
    {
        return scalar * a;
    }

    template <class T, std::size_t size, class NumberType>
    auto operator/(const AlgebraicArray<T, size>& a, NumberType scalar)
    {
        AlgebraicArray<T, size> a2;
        for (std::size_t i = 0; i < size; ++i)
        {
            a2[i] = a[i] / scalar;
        }
        return a2;
    }

    template <class NumberType, class T, std::size_t size>
    auto operator+(NumberType scalar, const AlgebraicArray<T, size>& a)
    {
        AlgebraicArray<T, size> a2;
        for (std::size_t i = 0; i < size; ++i)
        {
            a2[i] = scalar + a[i];
        }
        return a2;
    }

    template <class T, std::size_t size, class NumberType>
    auto operator+(const AlgebraicArray<T, size>& a, NumberType scalar)
    {
        return scalar + a;
    }

    template <class T, std::size_t size>
    auto operator+(const AlgebraicArray<T, size>& a, const AlgebraicArray<T, size>& b)
    {
        AlgebraicArray<T, size> c;
        for (std::size_t i = 0; i < size; ++i)
        {
            c[i] = a[i] + b[i];
        }
        return c;
    }

    template <class T, std::size_t size>
    auto operator-(const AlgebraicArray<T, size>& a, const AlgebraicArray<T, size>& b)
    {
        AlgebraicArray<T, size> c;
        for (std::size_t i = 0; i < size; ++i)
        {
            c[i] = a[i] - b[i];
        }
        return c;
    }

    template <class T, std::size_t size>
    auto operator*(const AlgebraicArray<T, size>& a, const AlgebraicArray<T, size>& b)
    {
        AlgebraicArray<T, size> c;
        for (std::size_t i = 0; i < size; ++i)
        {
            c[i] = a[i] * b[i];
        }
        return c;
    }

    template <class T, std::size_t size>
    auto operator/(const AlgebraicArray<T, size>& a, const AlgebraicArray<T, size>& b)
    {
        AlgebraicArray<T, size> c;
        for (std::size_t i = 0; i < size; ++i)
        {
            c[i] = a[i] / b[i];
        }
        return c;
    }

    template <class T, std::size_t size>
    auto operator-(const AlgebraicArray<T, size>& a)
    {
        return -1. * a;
    }

    template <class T, std::size_t size, class NumberType>
    auto pow(const AlgebraicArray<T, size>& a, NumberType exponent)
    {
        AlgebraicArray<T, size> b;
        for (std::size_t i = 0; i < size; ++i)
        {
            b[i] = std::pow(a[i], exponent);
        }
        return b;
    }

    template <class NumberType1,
              class NumberType2,
              typename std::enable_if_t<std::is_arithmetic_v<NumberType1> && std::is_arithmetic_v<NumberType2>, bool> = true>
    inline auto pow(NumberType1 base, NumberType2 exponent)
    {
        return std::pow(base, exponent);
    }

} // end namespace samurai
