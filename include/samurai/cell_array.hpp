// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>

#include <fmt/color.h>

#include "algorithm.hpp"
#include "cell_list.hpp"
#include "level_cell_array.hpp"
#include "utils.hpp"

namespace samurai
{

    //////////////////////////
    // CellArray definition //
    //////////////////////////

    /** @class CellArray
     *  @brief Array of LevelCellArray.
     *
     *  A box is defined by its minimum and maximum corners.
     *
     *  @tparam dim_ The dimension
     *  @tparam TInterval The type of the intervals (default type is Interval<int>).
     *  @tparam max_size_ The size of the array and the maximum levels (default size is 16).
     */
    template<std::size_t dim_, class TInterval=Interval<int>, std::size_t max_size_ = 16>
    class CellArray
    {
    public:
        static constexpr auto dim = dim_;
        static constexpr auto max_size = max_size_;

        using interval_t = TInterval;
        using lca_type = LevelCellArray<dim, TInterval>;
        using cl_type = CellList<dim, TInterval, max_size>;

        CellArray();
        CellArray(const cl_type& cl, bool with_update_index = true);

        const lca_type& operator[](std::size_t i) const;
        lca_type& operator[](std::size_t i);

        template<typename... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, T... index) const;

        std::size_t nb_cells() const;
        std::size_t nb_cells(std::size_t level) const;

        std::size_t max_level() const;
        std::size_t min_level() const;

        void update_index();

        void to_stream(std::ostream& os) const;

    private:
        std::array<lca_type, max_size + 1> m_cells;
    };

    //////////////////////////////
    // CellArray implementation //
    //////////////////////////////

    /**
     * Default contructor which sets the level for each LevelCellArray.
     */
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline CellArray<dim_, TInterval, max_size_>::CellArray()
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            m_cells[level] = {level};
        }
    }

    /**
     * Construction of a CellArray from a CellList
     *
     * @param cl The cell list.
     * @parma with_update_index A boolean indicating if the index of the x-intervals must be computed.
     */
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline CellArray<dim_, TInterval, max_size_>::CellArray(const cl_type& cl, bool with_update_index)
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            m_cells[level] = cl[level];
        }

        if (with_update_index)
        {
            update_index();
        }
    }

    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::operator[](std::size_t i) const -> const lca_type&
    {
        return m_cells[i];
    }

    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellArray<dim_, TInterval, max_size_>::operator[](std::size_t i) -> lca_type&
    {
        return m_cells[i];
    }

    /**
     * Return the x-interval satisfying the input parameters
     *
     * @param level The desired level.
     * @param interval The desired x-interval.
     * @param index The desired indices for the other dimensions.
     */
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    template<typename... T>
    inline auto CellArray<dim_, TInterval, max_size_>::get_interval(std::size_t level, const interval_t& interval, T... index) const -> const interval_t&
    {
        auto row = find(m_cells[level], {interval.start, index...});
        return m_cells[level][0][static_cast<std::size_t>(row)];
    }

    /**
     * Return the number of cells which is the sum of each x-interval size
     * over the levels.
     */
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::nb_cells() const
    {
        std::size_t size = 0;
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            size += m_cells[level].nb_cells();
        }
        return size;
    }

    /**
     * Return the number of cells which is the sum of each x-interval size
     * for a given level.
     *
     * @param level The level where to compute the number of cells
     */
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::nb_cells(std::size_t level) const
    {
        return m_cells[level].nb_cells();
    }

    /**
     * Return the maximum level where the array entry is not empty.
     */
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::max_level() const
    {
        for (std::size_t level = max_size; level != std::size_t(-1); --level)
        {
            if (!m_cells[level].empty())
            {
                return level;
            }
        }
        return 0;
    }

    /**
     * Return the minimum level where the array entry is not empty.
     */
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::size_t CellArray<dim_, TInterval, max_size_>::min_level() const
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            if (!m_cells[level].empty())
            {
                return level;
            }
        }
        return max_size + 1;
    }

    /**
     * Update the index in the x-intervals allowing to navigate in the
     * Field data structure.
     */
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline void CellArray<dim_, TInterval, max_size_>::update_index()
    {
        using index_t = typename interval_t::index_t;
        std::size_t acc_size = 0;
        for_each_interval(*this, [&](auto, auto& interval, auto)
        {
            interval.index = safe_subs<index_t>(acc_size, interval.start);
            acc_size += interval.size();
        });
    }

    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline void CellArray<dim_, TInterval, max_size_>::to_stream(std::ostream& os) const
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            if (!m_cells[level].empty())
            {
                os <<
                        fmt::format(fg(fmt::color::steel_blue) | fmt::emphasis::bold,
                        "┌{0:─^{2}}┐\n"
                        "│{1: ^{2}}│\n"
                        "└{0:─^{2}}┘\n", "", fmt::format("Level {}", level), 20);
                m_cells[level].to_stream(os);
                os << std::endl;
            }
        }
    }

    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::ostream& operator<<(std::ostream& out,
                                    const CellArray<dim_, TInterval, max_size_>& cell_array)
    {
        cell_array.to_stream(out);
        return out;
    }

    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    inline bool operator==(const CellArray<dim_, TInterval, max_size_> &ca1, const CellArray<dim_, TInterval, max_size_>& ca2)
    {
        if (ca1.max_level() != ca2.max_level() ||
            ca1.min_level() != ca2.min_level())
        {
            return false;
        }

        for(std::size_t level=ca1.min_level(); level <= ca1.max_level(); ++level)
        {
            if (!(ca1[level] == ca2[level]))
            {
                return false;
            }
        }
        return true;
    }

} // namespace samurai