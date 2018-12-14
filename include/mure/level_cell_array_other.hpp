#pragma once

#include <array>
#include <type_traits>
#include <utility>
#include <ostream>
#include <deque>

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
    /// Construct from a level cell list
    inline LevelCellArray(LevelCellList<MRConfig> const& lcl);

    /// Display to the given stream
    void to_stream(std::ostream& out) const;

    /// Apply a given function to each interval along x
    template <typename TFunction>
    void for_each_interval_in_x(TFunction && f) const;

private:
    /// Recursive construction from a level cell list along dimension > 0
    template <std::size_t N>
    inline void initFromLevelCellList(LevelCellList<MRConfig> const& lcl,
                                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index,
                                      std::array<std::deque<interval_t>, dim> & cells,
                                      std::array<std::deque<index_t>, dim-1> & offsets,
                                      std::integral_constant<std::size_t, N>);

    /// Recursive construction from a level cell deque for the dimension 0
    inline void initFromLevelCellList(LevelCellList<MRConfig> const& lcl,
                                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index,
                                      std::array<std::deque<interval_t>, dim> & cells,
                                      std::array<std::deque<index_t>, dim-1> & offsets,
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

private:
    std::array<std::vector<interval_t>, dim> m_cells;   ///< All intervals in every direction
    std::array<std::vector<index_t>, dim-1>  m_offsets; ///< Offsets in interval list for each dim > 0
    Box<coord_index_t, dim-1> m_box_yz;                 ///< Bounding box along dimensions > 0
};


template <class MRConfig>
LevelCellArray<MRConfig>::LevelCellArray(LevelCellList<MRConfig> const &lcl)
{
    // Temporaries cells and offsets using list so that appending is more efficient
    std::array<std::deque<interval_t>, dim> cells;
    std::array<std::deque<index_t>, dim-1>  offsets;

    // Filling cells and offsets from the level cell list
    initFromLevelCellList(lcl, {}, cells, offsets, std::integral_constant<std::size_t, dim-1>{});

    // Flattening cells
    for (std::size_t N = 0; N < dim; ++N)
    {
        m_cells[N].reserve(cells[N].size());
        m_cells[N].insert(m_cells[N].begin(), cells[N].cbegin(), cells[N].cend());
    }

    // Flattening offsets
    for (std::size_t N = 0; N < dim-1; ++N)
    {
        m_offsets[N].reserve(offsets[N].size() + 1);
        m_offsets[N].insert(m_offsets[N].begin(), offsets[N].cbegin(), offsets[N].cend());
    }

    // Additionnal offset so that [m_offset[i], m_offset[i+1][ is always valid.
    for (std::size_t N = 0; N < dim-1; ++N)
        m_offsets[N].push_back(m_cells[N].size());

    // Updating bounding box along dimension > 0 (could be done in initFromCellList)
    if (dim > 1)
    {
        m_box_yz.min_corner().fill(std::numeric_limits<coord_index_t>::max());
        m_box_yz.max_corner().fill(std::numeric_limits<coord_index_t>::min());
        for (std::size_t N = 1; N < dim; ++N)
        {
            for (auto const& interval : m_cells[N])
            {
                m_box_yz.min_corner()[N-1] = std::min(m_box_yz.min_corner()[N-1], interval.start);
                m_box_yz.max_corner()[N-1] = std::max(m_box_yz.max_corner()[N-1], interval.end);
            }
        }
    }
}

template <class MRConfig>
template <std::size_t N>
void
LevelCellArray<MRConfig>::
initFromLevelCellList(LevelCellList<MRConfig> const& lcl,
                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index,
                      std::array<std::deque<interval_t>, dim> & cells,
                      std::array<std::deque<index_t>, dim-1> & offsets,
                      std::integral_constant<std::size_t, N>)
{
    // Working interval
    interval_t curr_interval(0, 0, offsets[N-1].size());

    // For each position along the Nth dimension
    for (coord_index_t i = lcl.min_corner_yz()[N-1]; i < lcl.max_corner_yz()[N-1]; ++i)
    {
        // Recursive call on the current position for the (N-1)th dimension
        index[N-1] = i;
        const std::size_t previous_offset = cells[N-1].size();
        initFromLevelCellList(lcl, index, cells, offsets, std::integral_constant<std::size_t, N-1>{});

        // If the co-dimensions are empty
        if (cells[N-1].size() == previous_offset)
        {
            // Adding the working interval if valid
            if (curr_interval.is_valid())
            {
                cells[N].push_back(curr_interval);
                curr_interval = interval_t{0, 0, 0};
            }
        }
        else // Co-dimensions are not empty
        {
            // Creating or updating the current interval
            if (curr_interval.is_valid())
                curr_interval.end = i+1;
            else
                curr_interval = interval_t(i, i+1, offsets[N-1].size() - i); // So that coord+index points to the data

            // Updating offsets
            offsets[N-1].push_back(previous_offset);
        }
    }

    // Adding the working interval if valid
    if (curr_interval.is_valid())
        cells[N].push_back(curr_interval);
}

template <class MRConfig>
void
LevelCellArray<MRConfig>::
initFromLevelCellList(LevelCellList<MRConfig> const& lcl,
                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index,
                      std::array<std::deque<interval_t>, dim> & cells,
                      std::array<std::deque<index_t>, dim-1> & offsets,
                      std::integral_constant<std::size_t, 0>)
{
    // Along the X axis, simply copy the intervals in cells[0]
    auto const& interval_list = lcl[index];
    cells[0].insert(cells[0].end(), interval_list.begin(), interval_list.end());
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

    out << "Box\t" << m_box_yz.min_corner() << " " << m_box_yz.max_corner() << std::endl;
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

template <class MRConfig>
std::ostream& operator<< (std::ostream& out, LevelCellArray<MRConfig> const& level_cell_array)
{
    level_cell_array.to_stream(out);
    return out;
}

} // namespace mure
