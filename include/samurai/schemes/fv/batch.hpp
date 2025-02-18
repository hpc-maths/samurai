#pragma once
#include "../../stencil.hpp"
#include "std_vector_wrapper.hpp"

namespace samurai
{
    template <class T, std::size_t array_size>
    class ArrayBatch
    {
      public:

        using value_type = T;
        // using dynamic_vector_t = StdVectorWrapper<T>;
        using dynamic_vector_t = field_data_storage_t<T, 1>;
        //  using dynamic_vector_t = xt::xtensor<T, 1>;

      private:

        static constexpr std::size_t MAX_SIZE = 128;
        // using dynamic_vector_t = AlgebraicArray<T, MAX_SIZE>;

        std::array<dynamic_vector_t, array_size> m_batch;

        std::size_t m_position = 0;

      public:

        ArrayBatch()
        {
        }

        ArrayBatch(std::size_t batch_size)
        {
            resize(batch_size);
        }

        inline auto& batch()
        {
            return m_batch;
        }

        inline const auto& batch() const
        {
            return m_batch;
        }

        inline std::size_t capacity() const
        {
            return m_batch[0].size();
        }

        inline std::size_t size() const
        {
            return m_position;
        }

        inline const auto& position() const
        {
            return m_position;
        }

        inline auto& position()
        {
            return m_position;
        }

        inline void reset_position()
        {
            m_position = 0;
        }

        inline auto& operator[](std::size_t index_in_array)
        {
            return m_batch[index_in_array];
        }

        // inline auto operator[](std::size_t index_in_array)
        // {
        //     range_t<long long> range{0, static_cast<long long>(capacity() - 1)};
        //     return samurai::view(m_batch[index_in_array], range);
        // }

        inline const auto& operator[](std::size_t index_in_array) const
        {
            return m_batch[index_in_array];
        }

        // inline auto operator[](std::size_t index_in_array) const
        // {
        //     range_t<long long> range{0, static_cast<long long>(capacity() - 1)};
        //     return samurai::view(m_batch[index_in_array], range);
        // }

        inline void resize(std::size_t batch_size)
        {
            if constexpr (array_size == 1)
            {
                m_batch.resize(batch_size);
            }
            else
            {
                if constexpr (!std::is_same_v<dynamic_vector_t, StdArrayWrapper<T, MAX_SIZE>>)
                {
                    for (std::size_t i = 0; i < array_size; ++i)
                    {
                        // if constexpr (std::is_same_v<dynamic_vector_t, xt::xtensor<T, 1>>)
                        // {
                        //     m_batch[i].resize({batch_size});
                        // }
                        // else
                        // {
                        m_batch[i].resize(batch_size);
                        //}
                    }
                }
            }
            reset_position();
        }

        // inline void add(const CollapsStdArray<T, array_size>& values)
        inline void add(const std::array<T, array_size>& values)
        {
            if constexpr (array_size == 1)
            {
                m_batch[m_position].values;
            }
            else
            {
                for (std::size_t i = 0; i < array_size; ++i)
                {
                    m_batch[i][m_position] = values[i];
                }
            }
            m_position++;
        }

        template <class Func>
        inline void add(const std::array<T, array_size>& values, Func&& copy)
        {
            if constexpr (array_size == 1)
            {
                m_batch[m_position].values;
            }
            else
            {
                for (std::size_t i = 0; i < array_size; ++i)
                {
                    copy(m_batch[i][m_position], values[i]);
                }
            }
            m_position++;
        }

        // inline bool empty() const
        // {
        //     if constexpr (array_size == 1)
        //     {
        //         return m_batch[0].empty();
        //     }
        //     else
        //     {
        //         return m_batch[0].empty();
        //     }
        // }
    };

    template <class T>
    using Batch = StdVectorWrapper<T>;

    // template <class T1, class T2, std::size_t size, class Func>
    // void transform(const ArrayBatch<T1, size>& input, ArrayBatch<T2, size>& output, Func&& op)
    // {
    //     output.resize(input.size());
    //     for (std::size_t i = 0; i < size; ++i)
    //     {
    //         for (std::size_t j = 0; j < input.position(); ++j)
    //         {
    //             output[i][j] = op(input[i][j]);
    //         }
    //     }
    //     output.position() = input.position();
    // }

    template <class Mesh, std::size_t stencil_size, class Cell>
    void copy_to_batch(IteratorStencil<Mesh, stencil_size>& stencil_it, std::size_t length, ArrayBatch<Cell, stencil_size>& stencil_batch)
    {
        using index_t = typename Cell::index_t;

        auto start = stencil_batch.position();
        for (std::size_t s = 0; s < stencil_size; ++s)
        {
            for (std::size_t ii = 0; ii < length; ++ii)
            {
                // stencil_batch[s][start + ii].level      = m_cells[s].level;
                stencil_batch[s][start + ii].index = stencil_it.cells()[s].index + static_cast<index_t>(ii);
                // stencil_batch[s][start + ii].indices[0] = m_cells[s].indices[0] + static_cast<value_t>(ii);
            }
        }
        stencil_batch.position() += length;
        for (Cell& cell : stencil_it.cells())
        {
            cell.index += length;
            cell.indices[0] += length;
        }
    }

    template <std::size_t index_coarse_cell, class Mesh, std::size_t stencil_size, class Cell>
    void
    copy_to_batch(LevelJumpIterator<index_coarse_cell, Mesh, stencil_size>& stencil_it, std::size_t length, ArrayBatch<Cell, 2>& stencil_batch)
    {
        for (std::size_t ii = 0; ii < length; ++ii)
        {
            stencil_batch.add(stencil_it.cells(),
                              [](Cell& dest, const Cell& src)
                              {
                                  dest.index = src.index;
                                  // dest.indices = src.indices;
                              });
            stencil_it.move_next();
        }
    }

    template <class Mesh, std::size_t stencil_size, class T, class Field>
    void copy_values_to_batch(IteratorStencil<Mesh, stencil_size>& stencil_it,
                              std::size_t length,
                              ArrayBatch<T, stencil_size>& stencil_values_batch,
                              const Field& field)
    {
        auto start = stencil_values_batch.position();
        assert(start + length <= stencil_values_batch.capacity());
        for (std::size_t s = 0; s < stencil_size; ++s)
        {
            for (std::size_t ii = 0; ii < length; ++ii)
            {
                stencil_values_batch[s][start + ii] = field[static_cast<std::size_t>(stencil_it.cells()[s].index) + ii];
            }
        }
        stencil_values_batch.position() += length;
    }

} // end namespace samurai
