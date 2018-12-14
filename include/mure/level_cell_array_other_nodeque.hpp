#pragma once

#include <array>
#include <type_traits>
#include <utility>
#include <ostream>

#include "mure/interval.hpp"
#include "mure/interval_value_range_no_boost.hpp"
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
                                      std::integral_constant<std::size_t, N>);

    /// Recursive construction from a level cell list for the dimension 0
    inline void initFromLevelCellList(LevelCellList<MRConfig> const& lcl,
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

private:
    std::array<std::vector<interval_t>, dim> m_cells;     ///< All intervals in every direction
    std::array<std::vector<index_t>, dim-1>  m_offsets;   ///< Offsets in interval list for each dim > 1
};


template <class MRConfig>
LevelCellArray<MRConfig>::LevelCellArray(LevelCellList<MRConfig> const &lcl)
{
    // Estimating reservation size
    std::size_t cnt_x = 0;
    std::size_t cnt_yz = 0;

    for (auto const& l : lcl.m_grid_yz)
    {
        if (l.size() > 0)
        {
            cnt_x += l.size();
            ++cnt_yz;
        }
    }

    std::size_t size = 1;
    for (std::size_t N = dim-1; N >= 1; --N)
    {
        size *= lcl.max_corner_yz()[N-1] - lcl.min_corner_yz()[N-1];
        m_cells[N].reserve(std::min(size/2, cnt_yz));
        m_offsets[N-1].reserve(std::min(size, cnt_yz)+1);
    }
    m_cells[0].reserve(cnt_x);

    std::cout << "m_cells capacity: ";
    for (auto const& c : m_cells)
        std::cout << c.capacity() << " ";
    std::cout << std::endl;
    
    std::cout << "m_offsets capacity: ";
    for (auto const& c : m_offsets)
        std::cout << c.capacity() << " ";
    std::cout << std::endl;

    // Filling cells and offsets from the level cell list
    initFromLevelCellList(lcl, {}, std::integral_constant<std::size_t, dim-1>{});

    // Additionnal offset so that [m_offset[i], m_offset[i+1][ is always valid.
    for (std::size_t N = 0; N < dim-1; ++N)
        m_offsets[N].push_back(m_cells[N].size());

    // Adjusting capacity
    for (auto& c : m_cells)
        c.reserve(c.size());
    
    for (auto& c : m_offsets)
        c.reserve(c.size());
    
    std::cout << "m_cells size: ";
    for (auto const& c : m_cells)
        std::cout << c.size() << " ";
    std::cout << std::endl;
    
    std::cout << "m_offsets size: ";
    for (auto const& c : m_offsets)
        std::cout << c.size() << " ";
    std::cout << std::endl;
}

template <class MRConfig>
template <std::size_t N>
void
LevelCellArray<MRConfig>::
initFromLevelCellList(LevelCellList<MRConfig> const& lcl,
                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index,
                      std::integral_constant<std::size_t, N>)
{
    // Working interval
    interval_t curr_interval(0, 0, m_offsets[N-1].size());

    // For each position along the Nth dimension
    for (coord_index_t i = lcl.min_corner_yz()[N-1]; i < lcl.max_corner_yz()[N-1]; ++i)
    {
        // Recursive call on the current position for the (N-1)th dimension
        index[N-1] = i;
        const std::size_t previous_offset = m_cells[N-1].size();
        initFromLevelCellList(lcl, index, std::integral_constant<std::size_t, N-1>{});

        // If the co-dimensions are empty
        if (m_cells[N-1].size() == previous_offset)
        {
            // Adding the working interval if valid
            if (curr_interval.is_valid())
            {
                m_cells[N].push_back(curr_interval);
                curr_interval = interval_t{0, 0, 0};
            }
        }
        else // Co-dimensions are not empty
        {
            // Creating or updating the current interval
            if (curr_interval.is_valid())
                curr_interval.end = i+1;
            else
                curr_interval = interval_t(i, i+1, m_offsets[N-1].size() - i);

            // Updating m_offsets
            m_offsets[N-1].push_back(previous_offset);
        }
    }

    // Adding the working interval if valid
    if (curr_interval.is_valid())
        m_cells[N].push_back(curr_interval);
}

template <class MRConfig>
void
LevelCellArray<MRConfig>::
initFromLevelCellList(LevelCellList<MRConfig> const& lcl,
                      xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index,
                      std::integral_constant<std::size_t, 0>)
{
    // Along the X axis, simply copy the intervals in cells[0]
    auto const& interval_list = lcl[index];
    m_cells[0].insert(m_cells[0].end(), interval_list.begin(), interval_list.end());
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

template <class MRConfig>
std::ostream& operator<< (std::ostream& out, LevelCellArray<MRConfig> const& level_cell_array)
{
    level_cell_array.to_stream(out);
    return out;
}

} // namespace mure
