#pragma once

#include <array>
#include <type_traits>
#include <utility>
#include <ostream>

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include "mure/box.hpp"
#include "mure/cell.hpp"
#include "mure/interval.hpp"
#include "mure/level_cell_list.hpp"

namespace mure
{

template <class MRConfig>
class LevelCellArray
{
public:
    constexpr static auto dim = MRConfig::dim;
    using index_t       = typename MRConfig::index_t;
    using coord_index_t = typename MRConfig::coord_index_t;
    using interval_t    = Interval<coord_index_t, index_t>;

public:
    LevelCellArray(LevelCellArray&&) = default;
    LevelCellArray& operator=(LevelCellArray&&) = default;

    LevelCellArray(LevelCellArray const&) = default;
    LevelCellArray& operator=(LevelCellArray const&) = default;

    /// Construct from a level cell list
    inline LevelCellArray(LevelCellList<MRConfig> const& lcl = {});

    inline LevelCellArray(Box<coord_index_t, dim> box);

    /// Display to the given stream
    void to_stream(std::ostream& out) const;

    /// Apply a given function to each interval along x
    template <typename TFunction>
    void for_each_interval_in_x(TFunction && f) const;

    //// checks whether the container is empty
    inline bool empty() const;

    //// Get the minimum corner in [y, z] where
    //// all the coordinates in y and z are included
    inline auto min_corner_yz() const;

    //// Get the maximum corner in [y, z] where
    //// all the coordinates in y and z are included
    inline auto max_corner_yz() const;

    //// Gives the number of intervals in each dimension
    inline auto shape() const;

    //// Gives the number of cells
    inline auto nb_cells() const;

    auto const& operator[](index_t d) const;
    auto& operator[](index_t d);

    //// Return a const reference to the offsets array
    //// for dimension d
    auto const& offsets(index_t d) const;

    //// Return a reference to the offsets array
    //// for dimension d
    auto& offsets(index_t d);

    //// Apply a function on each cell
    template<class TFunction>
    void for_each_cell(TFunction&& func, std::size_t level) const;

    //// Find the interval in x for the indices (ix, iy, iz)
    //// return -1 if not found
    inline int find(xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord) const;

    //// Apply a function on each cell
    template<class TFunction>
    void for_each_block(TFunction&& func) const;

private:
    /// Recursive construction from a level cell list along dimension > 0
    template <typename TGrid, std::size_t N>
    inline void initFromLevelCellList(TGrid const& grid,
                                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index,
                                      std::integral_constant<std::size_t, N>);

    /// Recursive construction from a level cell list for the dimension 0
    template <typename TIntervalList>
    inline void initFromLevelCellList(TIntervalList const& interval_list,
                                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index,
                                      std::integral_constant<std::size_t, 0>);

    /// Recursive apply of a function on each interval along x, for dimension > 0
    template <typename TFunction, std::size_t N>
    inline void for_each_interval_in_x_impl(TFunction && f,
                                            xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index,
                                            index_t start_index, index_t end_index,
                                            std::integral_constant<std::size_t, N>) const;

    /// Recursive apply of a function on each interval along x, for the dimension 0
    template <typename TFunction>
    inline void for_each_interval_in_x_impl(TFunction && f,
                                            xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index,
                                            index_t start_index, index_t end_index,
                                            std::integral_constant<std::size_t, 0>) const;

    /// Recursive apply of a function on each cell, for the dimension > 0
    template <typename TFunction, std::size_t N>
    inline void for_each_cell_impl(TFunction&& func, std::size_t level,
                                   xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                                   index_t start_index, index_t end_index,
                                   std::integral_constant<std::size_t, N>) const;

    /// Recursive apply of a function on each cell, for the dimension 0
    template <typename TFunction>
    inline void for_each_cell_impl(TFunction&& func, std::size_t level,
                                   xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                                   index_t start_index, index_t end_index,
                                   std::integral_constant<std::size_t, 0>) const;

    template <std::size_t N>
    inline int find_impl(index_t start_index, index_t end_index,
                         xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord,
                         std::integral_constant<std::size_t, N>) const;

    inline int find_impl(index_t start_index, index_t end_index,
                         xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord,
                         std::integral_constant<std::size_t, 0>) const;

    template <typename TFunction, std::size_t N>
    inline void for_each_block_impl(TFunction&& func,
                                    index_t start_index, index_t end_index,
                                    xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                                    std::integral_constant<std::size_t, N>) const;

    template <typename TFunction>
    inline void for_each_block_impl(TFunction&& func,
                                    index_t start_index, index_t end_index,
                                    xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                                    std::integral_constant<std::size_t, 0>) const;

private:
    std::array<std::vector<interval_t>, dim> m_cells;     ///< All intervals in every direction
    std::array<std::vector<index_t>, dim-1>  m_offsets;   ///< Offsets in interval list for each dim > 1
};


template <class MRConfig>
LevelCellArray<MRConfig>::LevelCellArray(LevelCellList<MRConfig> const &lcl)
{
    /* Estimating reservation size
     *
     * NOTE: the estimation takes time, more than the time needed for reallocating the vectors...
     * Maybe 2 other solutions:
     * - (highly) overestimating the needed size since the memory will be actually allocated only when touched (at least under Linux)
     * - cnt_x and cnt_yz updated in LevelCellList during the filling process
     *
     * NOTE2: in fact, hard setting the optimal values for cnt_x and cnt_yz doesn't speedup things, strang...
     */

    // Filling cells and offsets from the level cell list
    initFromLevelCellList(lcl.grid_yz(), {}, std::integral_constant<std::size_t, dim-1>{});

    // Additionnal offset so that [m_offset[i], m_offset[i+1][ is always valid.
    for (std::size_t N = 0; N < dim-1; ++N)
        m_offsets[N].push_back(m_cells[N].size());
}

template <class MRConfig>
LevelCellArray<MRConfig>::LevelCellArray(Box<coord_index_t, dim> box)
{
    auto dimensions = box.length();
    auto start = box.min_corner();
    auto end = box.max_corner();

    std::size_t size = 1;
    for(std::size_t d=dim-1; d>0; --d)
    {
        m_offsets[d-1].resize((dimensions[d]*size)+1);
        for(std::size_t i=0; i<(dimensions[d]*size)+1; ++i)
            m_offsets[d-1][i] = i;
        m_cells[d].resize(size);
        for(std::size_t i=0; i<size; ++i)
            m_cells[d][i] = {start[d], end[d], m_offsets[d-1][i*dimensions[d]] - start[d]};
        size *= dimensions[d];
    }

    m_cells[0].resize(size);
    for(std::size_t i=0; i<size; ++i)
        // m_cells[0][i] = {start[0], end[0], 0};
        m_cells[0][i] = {start[0], end[0], i*dimensions[0] - start[0]};
}

template <class MRConfig>
inline bool
LevelCellArray<MRConfig>::
empty() const
{
    return m_cells[0].empty();
}

template <class MRConfig>
inline auto
LevelCellArray<MRConfig>::
shape() const
{
    xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> output;
    for(std::size_t i=0; i<dim; ++i)
    {
        output[i] = m_cells[i].size();
    }
    return output;
}

template <class MRConfig>
inline auto
LevelCellArray<MRConfig>::
nb_cells() const
{
    auto op = [](auto&& init, auto const& interval){return std::move(init) + interval.size();};
    return std::accumulate(m_cells[0].cbegin(), m_cells[0].cend(), 0, op);
}

template <class MRConfig>
auto const&
LevelCellArray<MRConfig>::
operator[](index_t d) const
{
    return m_cells[d];
}

template <class MRConfig>
auto&
LevelCellArray<MRConfig>::
operator[](index_t d)
{
    return m_cells[d];
}

template <class MRConfig>
auto const&
LevelCellArray<MRConfig>::
offsets(index_t d) const
{
    assert(d > 0);
    return m_offsets[d-1];
}

template <class MRConfig>
auto&
LevelCellArray<MRConfig>::
offsets(index_t d)
{
    assert(d > 0);
    return m_offsets[d-1];
}

template <class MRConfig>
template <typename TGrid, std::size_t N>
void
LevelCellArray<MRConfig>::
initFromLevelCellList(TGrid const& grid,
                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index,
                      std::integral_constant<std::size_t, N>)
{
    // Working interval
    interval_t curr_interval(0, 0, 0);

    // For each position along the Nth dimension
    for (auto const& point : grid)
    {
        // Coordinate along the Nth dimension
        const auto i = point.first;

        // Recursive call on the current position for the (N-1)th dimension
        index[N-1] = i;
        const std::size_t previous_offset = m_cells[N-1].size();
        initFromLevelCellList(point.second, index, std::integral_constant<std::size_t, N-1>{});

        /* Since we move on a sparse storage, each coordinate have non-empty co-dimensions
         * So the question is, are we continuing an existing interval or have we jump to another one.
         *
         * WARNING: we are supposing that the sparse array of dimension dim-1 has no empty entry.
         *      Otherwise, we should check that the recursive call has do something by comparing
         *      previous_offset with the size of m_cells[N-1].
         */
        if (curr_interval.is_valid())
        {
            // If the coordinate has jump out of the current interval
            if (i > curr_interval.end)
            {
                // Adding the previous interval...
                m_cells[N].push_back(curr_interval);

                // ... and creating a new one.
                curr_interval = interval_t(i, i+1, m_offsets[N-1].size() - i);
            }
            else
            {
                // Otherwise, we are just continuing the current interval
                ++curr_interval.end;
            }
        }
        else
        {
            // If there is no current interval (at the beginning of the loop)
            // we create a new one.
            curr_interval = interval_t(i, i+1, m_offsets[N-1].size() - i);
        }

        // Updating m_offsets (at each iteration since we are always updating an interval)
        m_offsets[N-1].push_back(previous_offset);
    }

    // Adding the working interval if valid
    if (curr_interval.is_valid())
        m_cells[N].push_back(curr_interval);
}

template <class MRConfig>
template <typename TIntervalList>
void
LevelCellArray<MRConfig>::
initFromLevelCellList(TIntervalList const& interval_list,
                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index,
                      std::integral_constant<std::size_t, 0>)
{
    // Along the X axis, simply copy the intervals in cells[0]
    std::copy(interval_list.begin(), interval_list.end(), std::back_inserter(m_cells[0]));
}

template <class MRConfig>
void
LevelCellArray<MRConfig>::
to_stream(std::ostream& out) const
{
    for (std::size_t d = 0; d < dim; ++d)
    {
        out << "Dim " << d << std::endl;

        out << "\tcells = ";
        for (auto const& interval : m_cells[d])
            std::cout << interval << " ";
        out << std::endl;

        if (d > 0)
        {
            out << "\toffsets = ";
            for (auto const& v : m_offsets[d-1])
                std::cout << v << " ";
            out << std::endl;
        }
    }
}

template <typename MRConfig>
template <typename TFunction>
void
LevelCellArray<MRConfig>::
for_each_interval_in_x(TFunction && f) const
{
    for_each_interval_in_x_impl(std::forward<TFunction>(f),
                                {},
                                0,
                                m_cells[dim-1].size(),
                                std::integral_constant<std::size_t, dim-1>{});
}

template <typename MRConfig>
template <typename TFunction, std::size_t N>
void
LevelCellArray<MRConfig>::
for_each_interval_in_x_impl(TFunction && f,
                            xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index,
                            index_t start_index, index_t end_index,
                            std::integral_constant<std::size_t, N>) const
{
    for (index_t i = start_index; i < end_index; ++i)
    {
        auto const& interval = m_cells[N][i];
        for (coord_index_t c = interval.start; c < interval.end; ++c)
        {
            index[N-1] = c;
            for_each_interval_in_x_impl(std::forward<TFunction>(f),
                                        index,
                                        m_offsets[N-1][interval.index + c],
                                        m_offsets[N-1][interval.index + c + 1],
                                        std::integral_constant<std::size_t, N-1>{});
        }
    }
}

template <typename MRConfig>
template <typename TFunction>
void
LevelCellArray<MRConfig>::
for_each_interval_in_x_impl(TFunction && f,
                            xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index,
                            index_t start_index, index_t end_index,
                            std::integral_constant<std::size_t, 0>) const
{
    for (index_t i = start_index; i < end_index; ++i)
    {
        std::forward<TFunction>(f)(index, m_cells[0][i]);
    }
}

template <typename MRConfig>
template <typename TFunction>
void
LevelCellArray<MRConfig>::
for_each_cell(TFunction && f, std::size_t level) const
{
    for_each_cell_impl(std::forward<TFunction>(f),
                       level,
                       {},
                       0,
                       m_cells[dim-1].size(),
                       std::integral_constant<std::size_t, dim-1>{});
}

template <typename MRConfig>
template <typename TFunction, std::size_t N>
void
LevelCellArray<MRConfig>::
for_each_cell_impl(TFunction && f, std::size_t level,
                            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                            index_t start_index, index_t end_index,
                            std::integral_constant<std::size_t, N>) const
{
    for (index_t i = start_index; i < end_index; ++i)
    {
        auto const& interval = m_cells[N][i];
        for (coord_index_t c = interval.start; c < interval.end; ++c)
        {
            index[N] = c;
            for_each_cell_impl(std::forward<TFunction>(f),
                                        level,
                                        index,
                                        m_offsets[N-1][interval.index + c],
                                        m_offsets[N-1][interval.index + c + 1],
                                        std::integral_constant<std::size_t, N-1>{});
        }
    }
}

template <typename MRConfig>
template <typename TFunction>
void
LevelCellArray<MRConfig>::
for_each_cell_impl(TFunction&& f, std::size_t level,
                   xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                   index_t start_index, index_t end_index,
                   std::integral_constant<std::size_t, 0>) const
{
    for (index_t i = start_index; i < end_index; ++i)
    {
        auto const& interval = m_cells[0][i];
        for (coord_index_t c = interval.start; c < interval.end; ++c)
        {
            index[0] = c;
            Cell<coord_index_t, index_t, dim> cell{level, index, interval.index + c};
            std::forward<TFunction>(f)(cell);
        }
    }
}

template <typename MRConfig>
int
LevelCellArray<MRConfig>::
find(xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord) const
{
    return find_impl(0, m_cells[dim-1].size(), coord,
                     std::integral_constant<std::size_t, dim-1>{});
}

template <typename MRConfig>
template <std::size_t N>
int
LevelCellArray<MRConfig>::
find_impl(index_t start_index, index_t end_index,
          xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord,
          std::integral_constant<std::size_t, N>) const
{
    int find_index = -1;
    for (index_t i = start_index; i < end_index; ++i)
    {
        auto const& interval = m_cells[N][i];
        if (interval.contains(coord[N]))
        {
            find_index = find_impl(m_offsets[N-1][interval.index + coord[N]],
                                   m_offsets[N-1][interval.index + coord[N] + 1],
                                   coord,
                                   std::integral_constant<std::size_t, N-1>{});
        }
    }
    return find_index;
}

template <typename MRConfig>
int
LevelCellArray<MRConfig>::
find_impl(index_t start_index, index_t end_index,
          xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>  coord,
          std::integral_constant<std::size_t, 0>) const
{
    for (index_t i = start_index; i < end_index; ++i)
    {
        auto const& interval = m_cells[0][i];
        if (interval.contains(coord[0]))
        {
            return i;
        }
    }
    return -1;
}

template <typename MRConfig>
template <typename TFunction>
void
LevelCellArray<MRConfig>::
for_each_block(TFunction&& func) const
{
    for_each_block_impl(std::forward<TFunction>(func),
                        0, m_cells[dim-1].size(), {},
                        std::integral_constant<std::size_t, dim-1>{});
}

template <typename MRConfig>
template <typename TFunction, std::size_t N>
void
LevelCellArray<MRConfig>::
for_each_block_impl(TFunction&& func,
                    index_t start_index, index_t end_index,
                    xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                    std::integral_constant<std::size_t, N>) const
{
    for (index_t i = start_index; i < end_index; ++i)
    {
        auto const& interval = m_cells[N][i];
        for (coord_index_t c = interval.start; c < interval.end; ++c)
        {
            index[N] = c;
            for_each_block_impl(std::forward<TFunction>(func),
                                m_offsets[N-1][interval.index + c],
                                m_offsets[N-1][interval.index + c + 1],
                                index,
                                std::integral_constant<std::size_t, N-1>{});
        }
    }
}

template <typename MRConfig>
template <typename TFunction>
void
LevelCellArray<MRConfig>::
for_each_block_impl(TFunction&& func,
                    index_t start_index, index_t end_index,
                    xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                    std::integral_constant<std::size_t, 0>) const
{
    for (index_t i = start_index; i < end_index; ++i)
    {
        auto const& interval = m_cells[0][i];
        index[0] = interval.start;
        std::vector<xt::xtensor<int, 1>> stencil;
        if (dim == 2)
        {
            stencil = {{0, -1}, {0, 0}, {0, 1}};
        }
        if (dim == 3)
        {
            stencil = {{0, -1, -1}, {0, 0, -1}, {0, 1, -1},
                       {0, -1,  0}, {0, 0,  0}, {0, 1,  0},
                       {0, -1,  1}, {0, 0,  1}, {0, 1,  1}};
        }
        
        xt::xtensor_fixed<int, xt::xshape<static_cast<size_t>(std::pow(3, dim-1))>> rows;
        for(size_t i=0; i<stencil.size(); ++i)
        {
            rows[i] = find(index+stencil[i]);
        }

        if (xt::all(rows > -1))
        {
            if (dim == 2)
            {   
                auto load_input = [&](auto const& array)
                {
                    auto t = xt::xtensor<int, 2>::from_shape({3, interval.size()});
                    for(size_t i = 0; i < t.shape()[0]; ++i)
                    {
                        auto const& interval_tmp = m_cells[0][rows[i]];
                        xt::view(t, i, xt::all()) = xt::view(array, xt::range(interval_tmp.index + interval.start, interval_tmp.index + interval.end));
                    }
                    return t;
                };

                auto load_output = [&](auto & array) -> decltype(auto)
                {
                    auto const& interval_tmp = m_cells[0][rows[1]];
                    return xt::view(array, xt::newaxis(), xt::range(interval_tmp.index + interval.start+1, interval_tmp.index + interval.end-1));
                };
                func(load_input, load_output);
            }
            if (dim == 3)
            {   
                auto load_input = [&](auto const& array)
                {
                    auto t = xt::xtensor<int, 3>::from_shape({3, 3, interval.size()});
                    for(size_t j = 0; j < t.shape()[1]; ++j)
                    {
                        for(size_t i = 0; i < t.shape()[0]; ++i)
                        {
                            auto const& interval_tmp = m_cells[0][rows[i + 3*j]];
                            xt::view(t, i, j, xt::all()) = xt::view(array, xt::range(interval_tmp.index + interval.start, interval_tmp.index + interval.end));
                        }
                    }
                    return t;
                };

                auto load_output = [&](auto & array) -> decltype(auto)
                {
                    auto const& interval_tmp = m_cells[0][rows[4]];
                    return xt::view(array, xt::newaxis(), xt::newaxis(), xt::range(interval_tmp.index + interval.start+1, interval_tmp.index + interval.end-1));
                };
                func(load_input, load_output);
            }
        }
    }
}

template <class MRConfig>
std::ostream& operator<< (std::ostream& out, LevelCellArray<MRConfig> const& level_cell_array)
{
    level_cell_array.to_stream(out);
    return out;
}

} // namespace mure
