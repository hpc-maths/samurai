#pragma once
#include <array>
#include <math.h>

namespace samurai
{
    /**
     * Simple array with algebraic operations
     */
    template <class T, std::size_t size>
    struct StdArrayWrapper
    {
        using value_type = T;

      private:

        std::array<T, size> _a;

      public:

        StdArrayWrapper()
        {
        }

        explicit StdArrayWrapper(const std::array<T, size>& a)
            : _a(a)
        {
        }

        explicit StdArrayWrapper(std::array<T, size>&& a)
            : _a(std::move(a))
        {
        }

        template <class T2>
        StdArrayWrapper(std::initializer_list<T2> list)
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
        StdArrayWrapper(xt::xfunction<T2...> xt) // cppcheck-suppress noExplicitConstructor
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                this->_a[i] = xt(i);
            }
        }

        template <class... T2>
        StdArrayWrapper(xt::xview<T2...> xt) // cppcheck-suppress noExplicitConstructor
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                this->_a[i] = xt(i);
            }
        }

        StdArrayWrapper(T value) // cppcheck-suppress noExplicitConstructor
        {
            _a.fill(value);
        }

        auto& array()
        {
            return _a;
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
        auto operator=(T value)
        {
            fill(value);
        }

        template <class xt>
        auto operator=(const xt& other)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] = other(i);
            }
        }

        auto& operator+=(const StdArrayWrapper<T, size>& other)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] += other._a[i];
            }
            return *this;
        }

        auto& operator*=(const StdArrayWrapper<T, size>& other)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] *= other._a[i];
            }
            return *this;
        }

        auto& operator-=(const StdArrayWrapper<T, size>& other)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] -= other._a[i];
            }
            return *this;
        }

        auto& operator/=(const StdArrayWrapper<T, size>& other)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] /= other._a[i];
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator+=(NumberType scalar)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] += scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator*=(NumberType scalar)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] *= scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator-=(NumberType scalar)
        {
            for (std::size_t i = 0; i < size; ++i)
            {
                _a[i] -= scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator/=(NumberType scalar)
        {
            for (std::size_t i = 0; i < size; ++i)
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

    template <class NumberType, class T, std::size_t size>
    auto operator*(NumberType scalar, const StdArrayWrapper<T, size>& a)
    {
        StdArrayWrapper<T, size> b(a);
        b *= scalar;
        return b;
    }

    template <class T, std::size_t size, class NumberType>
    auto operator*(const StdArrayWrapper<T, size>& a, NumberType scalar)
    {
        return scalar * a;
    }

    template <class NumberType, class T, std::size_t size>
    StdArrayWrapper<T, size>&& operator*(NumberType scalar, StdArrayWrapper<T, size>&& a)
    {
        a *= scalar;
        return std::move(a);
    }

    template <class T, std::size_t size, class NumberType>
    StdArrayWrapper<T, size>&& operator*(StdArrayWrapper<T, size>&& a, NumberType scalar)
    {
        a *= scalar;
        return std::move(a);
    }

    template <class T, std::size_t size, class NumberType>
    auto operator/(const StdArrayWrapper<T, size>& a, NumberType scalar)
    {
        StdArrayWrapper<T, size> b(a);
        b /= scalar;
        return b;
    }

    template <class NumberType, class T, std::size_t size>
    auto operator+(NumberType scalar, const StdArrayWrapper<T, size>& a)
    {
        StdArrayWrapper<T, size> b(a);
        b += scalar;
        return b;
    }

    template <class T, std::size_t size, class NumberType>
    auto operator+(const StdArrayWrapper<T, size>& a, NumberType scalar)
    {
        return scalar + a;
    }

    template <class T, std::size_t size>
    auto operator+(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        StdArrayWrapper<T, size> c(a);
        c += b;
        return c;
    }

    template <class T, std::size_t size>
    StdArrayWrapper<T, size>&& operator+(const StdArrayWrapper<T, size>& a, StdArrayWrapper<T, size>&& b)
    {
        b += a;
        return std::move(b);
    }

    template <class T, std::size_t size>
    StdArrayWrapper<T, size>&& operator+(StdArrayWrapper<T, size>&& a, const StdArrayWrapper<T, size>& b)
    {
        a += b;
        return std::move(a);
    }

    template <class T, std::size_t size>
    auto operator+(StdArrayWrapper<T, size>&& a, StdArrayWrapper<T, size>&& b)
    {
        a += b;
        return std::move(a);
    }

    template <class T, std::size_t size>
    auto operator-(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        StdArrayWrapper<T, size> c(a);
        c -= b;
        return c;
    }

    template <class T, std::size_t size>
    StdArrayWrapper<T, size>&& operator-(StdArrayWrapper<T, size>&& a, const StdArrayWrapper<T, size>& b)
    {
        a -= b;
        return std::move(a);
    }

    template <class T, std::size_t size>
    auto operator*(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        StdArrayWrapper<T, size> c(a);
        c *= b;
        return c;
    }

    template <class T, std::size_t size>
    auto operator/(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        StdArrayWrapper<T, size> c(a);
        c /= b;
        return c;
    }

    template <class T, std::size_t size>
    auto operator-(const StdArrayWrapper<T, size>& a)
    {
        return -1. * a;
    }

    template <class T, std::size_t size, class NumberType>
    auto pow(const StdArrayWrapper<T, size>& a, NumberType exponent)
    {
        StdArrayWrapper<T, size> b;
        for (std::size_t i = 0; i < size; ++i)
        {
            b[i] = std::pow(a[i], exponent);
        }
        return b;
    }

    template <class T, std::size_t size, class NumberType>
    StdArrayWrapper<T, size>&& pow(StdArrayWrapper<T, size>&& a, NumberType exponent)
    {
        for (std::size_t i = 0; i < size; ++i)
        {
            a[i] = std::pow(a[i], exponent);
        }
        return std::move(a);
    }

    template <class NumberType1,
              class NumberType2,
              typename std::enable_if_t<std::is_arithmetic_v<NumberType1> && std::is_arithmetic_v<NumberType2>, bool> = true>
    inline auto pow(NumberType1 base, NumberType2 exponent)
    {
        return std::pow(base, exponent);
    }

} // end namespace samurai
