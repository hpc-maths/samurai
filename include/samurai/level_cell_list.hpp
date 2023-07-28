// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <iostream>
#include <map>
#include <type_traits>

#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

#include "cell.hpp"
#include "list_of_intervals.hpp"
#include "samurai_config.hpp"

namespace samurai
{
    namespace detail
    {
        /// Type helper to create nested std::map with final interval list
        template <typename TCoord, typename TIntervalList, std::size_t N>
        struct PartialGrid
        {
            using next_type = typename PartialGrid<TCoord, TIntervalList, N - 1>::type;
            using type      = std::map<TCoord, next_type>;
        };

        template <typename TCoord, typename TIntervalList>
        struct PartialGrid<TCoord, TIntervalList, 0>
        {
            using type = TIntervalList;
        };

        /** Nested std::map accessor.
         *
         * It is separated from the LevelCellList in order to automatically
         * manage the constness of the context (avoid duplicated code).
         */
        template <typename GridYZ, typename Index>
        inline decltype(auto) access_grid_yz(GridYZ& grid_yz, const Index&, std::integral_constant<std::size_t, 0>)
        {
            // For the first dimension, we return the interval list
            return grid_yz;
        }

        template <typename GridYZ, typename Index, std::size_t dim>
        inline decltype(auto) access_grid_yz(GridYZ& grid_yz, const Index& index, std::integral_constant<std::size_t, dim>)
        {
            // For other dimensions, we dive into the nested std::map
            return access_grid_yz(grid_yz[index[dim - 1]], index, std::integral_constant<std::size_t, dim - 1>{});
        }
    } // namespace detail

    //////////////////////////////
    // LevelCellList definition //
    //////////////////////////////
    template <std::size_t Dim, class TInterval = default_config::interval_t>
    class LevelCellList
    {
      public:

        static constexpr auto dim = Dim;
        using interval_t          = TInterval;
        using index_t             = typename interval_t::index_t;
        using coord_index_t       = typename interval_t::coord_index_t;
        using index_yz_t          = xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>>;
        using list_interval_t     = ListOfIntervals<coord_index_t, index_t>;

        /// Sparse dim-1 array that points to the interval lists along the x
        /// axis.
        using grid_t = typename detail::PartialGrid<coord_index_t, list_interval_t, dim - 1>::type;

        LevelCellList();
        LevelCellList(std::size_t level);

        const list_interval_t& operator[](const index_yz_t& index) const;
        list_interval_t& operator[](const index_yz_t& index);

        const grid_t& grid_yz() const;

        std::size_t level() const;

        bool empty() const;

        void to_stream(std::ostream& os) const;

        void add_cell(const Cell<dim, interval_t>& cell);

      private:

        grid_t m_grid_yz; ///< Sparse dim-1 array that points to the interval
                          ///< lists along the x axis.
        std::size_t m_level;
    };

    //////////////////////////////////
    // LevelCellList implementation //
    //////////////////////////////////
    template <std::size_t Dim, class TInterval>
    inline LevelCellList<Dim, TInterval>::LevelCellList()
        : m_level{0}
    {
    }

    template <std::size_t Dim, class TInterval>
    inline LevelCellList<Dim, TInterval>::LevelCellList(std::size_t level)
        : m_level{level}
    {
    }

    /// Constant access to the interval list at given dim-1 coordinates
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellList<Dim, TInterval>::operator[](const index_yz_t& index) const -> const list_interval_t&
    {
        return detail::access_grid_yz(m_grid_yz, index, std::integral_constant<std::size_t, dim - 1>{});
    }

    /// Mutable access to the interval list at given dim-1 coordinates
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellList<Dim, TInterval>::operator[](const index_yz_t& index) -> list_interval_t&
    {
        return detail::access_grid_yz(m_grid_yz, index, std::integral_constant<std::size_t, dim - 1>{});
    }

    /// Underlying sparse array
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellList<Dim, TInterval>::grid_yz() const -> const grid_t&
    {
        return m_grid_yz;
    }

    template <std::size_t Dim, class TInterval>
    inline std::size_t LevelCellList<Dim, TInterval>::level() const
    {
        return m_level;
    }

    template <std::size_t Dim, class TInterval>
    inline bool LevelCellList<Dim, TInterval>::empty() const
    {
        return m_grid_yz.empty();
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellList<Dim, TInterval>::to_stream(std::ostream& os) const
    {
        os << "LevelCellList\n";
        os << "=============\n";
        os << "TODO\n";
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellList<Dim, TInterval>::add_cell(const Cell<dim, interval_t>& cell)
    {
        using namespace xt::placeholders;

        (*this)[xt::view(cell.indices, xt::range(1, _))].add_point(cell.indices[0]);
    }

    template <std::size_t Dim, class TInterval>
    inline std::ostream& operator<<(std::ostream& out, const LevelCellList<Dim, TInterval>& level_cell_list)
    {
        level_cell_list.to_stream(out);
        return out;
    }

} // namespace samurai
