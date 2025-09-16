// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

template<typename T, std::size_t N>
class FixedCapacityNonTrivialStorage  
{
    static_assert(not std::is_trivially_constructible<T>::value);
public:
    constexpr       T* data()       { return reinterpret_cast<      T*>(m_core.data()); }
    constexpr const T* data() const { return reinterpret_cast<const T*>(m_core.data()); }
private:
    std::array<std::byte, sizeof(T)*N> m_core;
};

#include <array>
#include <cassert>

template<typename T, std::size_t N>
class FixedCapacityArray
{
    using Storage = std::conditional_t< std::is_trivially_constructible<T>::value, std::array<T,N>, FixedCapacityNonTrivialStorage<T,N> >;
public:
    FixedCapacityArray() : m_end(m_storage.data()) {}

    FixedCapacityArray(const std::size_t size, const T& value = T()) requires(std::is_trivially_constructible<T>::value) 
    { 
       assert(size <= N);
       m_end = begin() + size;
       std::fill(begin(), end(), value);
    }

    template<std::size_t N2>
    FixedCapacityArray(const FixedCapacityArray<T,N2>& other) requires(N2 <= N) 
        : m_end(m_storage.data() + other.size()) 
    { 
        if constexpr (std::is_fundamental<T>::value)
        {
            std::copy(std::cbegin(other), std::cend(other), begin()); 
        }
        else
        {
            const T* srcIt = std::cbegin(other);
            for (T* dstIt = begin(); dstIt != end(); ++dstIt, ++srcIt)
            {
                std::construct_at(dstIt, *srcIt);
            }
        } 
    }

    ~FixedCapacityArray() { clear(); }

    T* begin() { return m_storage.data(); }
    T* end()   { return m_end;            }

    const T* begin() const { return m_storage.data(); }
    const T* end()   const { return m_end;            }
    
    const T& operator[](const std::size_t i) const { return *(begin() + i); }
          T& operator[](const std::size_t i)       { return *(begin() + i); }

    std::size_t size() const { return end() - begin(); }

    constexpr std::size_t capacity() const { return N; }

    template<typename... Args>
    void emplace_back(Args&&... args) 
    { 
        assert(size() < N); 
        std::construct_at(m_end, std::forward<Args>(args)...); 
        ++m_end; 
    }

    void push_back(const T& value) { emplace_back(value); }

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
        std::destroy(begin(), end()); 
        m_end = begin(); 
    }
private:
    Storage m_storage;
    T* m_end;
};
