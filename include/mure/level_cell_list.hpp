#pragma once

#include <iostream>
#include <map>
#include <type_traits>

#include <xtensor/xfixed.hpp>

#include "list_of_intervals.hpp"

namespace mure
{
    namespace details
    {
        /// Type helper to create nested std::map with final interval list
        template<typename TCoord, typename TIntervalList, std::size_t N>
        struct PartialGrid
        {
            using type =
                std::map<TCoord, typename PartialGrid<TCoord, TIntervalList,
                                                      N - 1>::type>;
        };

        template<typename TCoord, typename TIntervalList>
        struct PartialGrid<TCoord, TIntervalList, 0>
        {
            using type = TIntervalList;
        };

        /** Nested std::map accessor.
         *
         * It is separated from the LevelCellList in order to automatically
         * manage the constness of the context (avoid duplicated code).
         */
        template<typename GridYZ, typename Index>
        inline decltype(auto)
        access_grid_yz(GridYZ &grid_yz, Index const &,
                       std::integral_constant<std::size_t, 0>)
        {
            // For the first dimension, we return the interval list
            return grid_yz;
        }

        template<typename GridYZ, typename Index, std::size_t dim>
        inline decltype(auto)
        access_grid_yz(GridYZ &grid_yz, Index const &index,
                       std::integral_constant<std::size_t, dim>)
        {
            // For other dimensions, we dive into the nested std::map
            return access_grid_yz(
                grid_yz[index[dim - 1]], index,
                std::integral_constant<std::size_t, dim - 1>{});
        }
    }

    template<std::size_t Dim, class TInterval = Interval<int>>
    class LevelCellList {
      public:
        constexpr static auto dim = Dim;
        using interval_t = TInterval;
        using index_t = typename interval_t::index_t;
        using coord_index_t = typename interval_t::value_t;
        using list_interval_t = ListOfIntervals<coord_index_t, index_t>;

        /// Sparse dim-1 array that points to the interval lists along the x
        /// axis.
        using grid_t =
            typename details::PartialGrid<coord_index_t, list_interval_t,
                                          dim - 1>::type;

        inline LevelCellList(std::size_t level) : m_level{level}
        {}

        LevelCellList() = default;

        /// Constant access to the interval list at given dim-1 coordinates
        inline list_interval_t const &operator[](
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const &index)
            const
        {
            return details::access_grid_yz(
                m_grid_yz, index,
                std::integral_constant<std::size_t, dim - 1>{});
        }

        /// Mutable access to the interval list at given dim-1 coordinates
        inline list_interval_t &operator[](
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const &index)
        {
            return details::access_grid_yz(
                m_grid_yz, index,
                std::integral_constant<std::size_t, dim - 1>{});
        }

        /// Underlying sparse array
        inline grid_t const &grid_yz() const
        {
            return m_grid_yz;
        }

        inline std::size_t level() const
        {
            return m_level;
        }

        inline void set_level(std::size_t level)
        {
            m_level = level;
        }

        inline void to_stream(std::ostream &os) const
        {
            os << "LevelCellList\n";
            os << "=============\n";
            os << "TODO\n"; // TODO
        }

      private:
        grid_t m_grid_yz; ///< Sparse dim-1 array that points to the interval
                          ///< lists along the x axis.
        std::size_t m_level{0};
    };

    template<std::size_t Dim, class TInterval>
    inline std::ostream &
    operator<<(std::ostream &out,
               const LevelCellList<Dim, TInterval> &level_cell_list)
    {
        level_cell_list.to_stream(out);
        return out;
    }
}
