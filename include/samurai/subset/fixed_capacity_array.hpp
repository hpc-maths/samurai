// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <cassert>

template <typename T, std::size_t N>
class FixedCapacityArray
{
    using CoreElement = std::conditional_t<std::is_trivially_constructible<T>::value, T, std::aligned_storage_t<sizeof(T), alignof(T)>>;

  public:

    FixedCapacityArray() noexcept
        : m_end(begin())
    {
    }

    FixedCapacityArray(const std::size_t size, const T& value = T())
        requires(std::is_trivially_constructible<T>::value)
    {
        assert(size <= N);
        m_end = begin() + size;
        for (T* it = begin(); it != end(); ++it)
        {
            std::construct_at(it, value);
        }
    }

    FixedCapacityArray(const FixedCapacityArray& other)
    {
        copyFrom(std::cbegin(other), std::cend(other));
    }

    FixedCapacityArray(FixedCapacityArray&& other) noexcept
    {
        moveFrom(std::cbegin(other), std::cend(other));
        other.clear();
    }

    ~FixedCapacityArray()
    {
        clear();
    }

    FixedCapacityArray& operator=(const FixedCapacityArray& other)
    {
        if (this != std::addressof(other))
        {
            clear();
            copyFrom(std::cbegin(other), std::cend(other), begin());
        }
        return *this;
    }

    FixedCapacityArray& operator=(FixedCapacityArray&& other) noexcept
    {
        if (this != std::addressof(other))
        {
            clear();
            moveFrom(std::cbegin(other), std::cend(other), begin());
            other.clear();
        }
        return *this;
    }

    T* begin() noexcept
    {
        if constexpr (std::is_trivially_constructible<T>::value)
        {
            return m_core.data();
        }
        else
        {
            return reinterpret_cast<T*>(m_core.data());
        }
    }

    T* end()
    {
        return m_end;
    }

    const T* begin() const noexcept
    {
        if constexpr (std::is_trivially_constructible<T>::value)
        {
            return m_core.data();
        }
        else
        {
            return reinterpret_cast<const T*>(m_core.data());
        }
    }

    const T* end() const
    {
        return m_end;
    }

    const T& operator[](const std::size_t i) const
    {
        return *(begin() + i);
    }

    T& operator[](const std::size_t i)
    {
        return *(begin() + i);
    }

    std::ptrdiff_t ssize() const
    {
        return std::distance(begin(), end());
    }

    std::size_t size() const
    {
        return std::size_t(ssize());
    }

    constexpr std::size_t capacity() const
    {
        return N;
    }

    template <typename... Args>
    void emplace_back(Args&&... args)
    {
        assert(size() < N);
        std::construct_at(m_end, std::forward<Args>(args)...);
        ++m_end;
    }

    void push_back(const T& value)
    {
        emplace_back(value);
    }

    void pop_back()
    {
        if (begin() != m_end)
        {
            --m_end;
            std::destroy_at(m_end);
        }
    }

    void clear()
    {
        if constexpr (not std::is_trivially_destructible_v<T>)
        {
            std::destroy(begin(), end());
        }
        m_end = begin();
    }

  private:

    void copyFrom(const T* srcBegin, const T* srcEnd)
    {
        if constexpr (std::is_trivially_copyable<T>::value)
        {
            std::copy(srcBegin, srcEnd, begin());
            m_end = begin() + std::distance(srcBegin, srcEnd);
        }
        else
        {
            m_end = begin();
            for (const T* srcIt = srcBegin; srcIt != srcEnd; ++srcIt, ++m_end)
            {
                std::construct_at(m_end, *srcIt);
            }
        }
    }

    void moveFrom(const T* srcBegin, const T* srcEnd)
    {
        if constexpr (std::is_trivially_move_constructible<T>::value)
        {
            std::move(srcBegin, srcEnd, begin());
            m_end = begin() + std::distance(srcBegin, srcEnd);
        }
        else
        {
            m_end = begin();
            for (const T* srcIt = srcBegin; srcIt != srcEnd; ++srcIt, ++m_end)
            {
                std::construct_at(m_end, std::move(*srcIt));
            }
        }
    }

    std::array<CoreElement, N> m_core;
    T* m_end;
};
