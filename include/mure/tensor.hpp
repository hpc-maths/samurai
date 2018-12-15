#pragma once

#include <vector>
#include <array>

/// nD tensor with column-order storage
template <
    class T,
    std::size_t N
>
class Tensor
{
public:
    static constexpr std::size_t dim = N;
    using value_type = T;

    Tensor()
    {
        m_shape.fill(0);
        m_size_cum.fill(0);
    }

    // TODO
    Tensor(Tensor const&) = delete;
    Tensor(Tensor &&) = delete;
    Tensor& operator= (Tensor const&) = delete;
    Tensor& operator= (Tensor &&) = delete;

    template <typename TShape>
    void resize(TShape const& shape)
    {
        // FIXME: no copy if data were already stored
        
        // Cumulative size (to speedup index calculation)
        m_size_cum[0] = shape[0];
        for (std::size_t i = 1; i < N; ++i)
            m_size_cum[i] = m_size_cum[i-1]*shape[i];

        m_data.resize(m_size_cum[N-1]);
    }

    template <typename TCoord>
    T& operator[] (TCoord const& coord) noexcept
    {
        return m_data[index_at(coord)];
    }
    
    template <typename TCoord>
    T const& operator[] (TCoord const& coord) const noexcept
    {
        return m_data[index_at(coord)];
    }

    auto begin()        noexcept { return m_data.begin(); }
    auto end()          noexcept { return m_data.end(); }
    auto begin()  const noexcept { return m_data.begin(); }
    auto end()    const noexcept { return m_data.end(); }
    auto cbegin() const noexcept { return m_data.cbegin(); }
    auto cend()   const noexcept { return m_data.cend(); }

private:
    template <typename TCoord>
    std::size_t index_at(TCoord const& coord) const noexcept
    {
        std::size_t index = coord[0];
        for (std::size_t i = 1; i < N; ++i) // Should be unrolled by compiler
            index += coord[i] * m_size_cum[i-1];
        return index;
    }

private:
    std::array<std::size_t, N> m_shape;
    std::array<std::size_t, N> m_size_cum;
    std::vector<T> m_data;

};
