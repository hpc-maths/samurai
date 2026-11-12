// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <iostream>
#ifdef SAMURAI_CELL_LIST_CONTAINER_LIST_OF_INTERVALS
#include <map>
#endif
#include <type_traits>
#include <utility>
#include <vector>

#include <xtensor/containers/xfixed.hpp>
#include <xtensor/views/xview.hpp>

#include "cell.hpp"
#include "list_of_intervals.hpp"
#include "samurai_config.hpp"

namespace samurai
{
    namespace detail
    {
        /// Sorted vector of (coordinate, value) pairs, used like a std::map.
        /// operator[] starts its search from the last row touched: rows are
        /// mostly visited in increasing order, so a new row is usually appended.
        template <typename TCoord, typename T>
        class FlatMap : private std::vector<std::pair<TCoord, T>>
        {
          public:

            using base_t = std::vector<std::pair<TCoord, T>>;
            using base_t::begin;
            using base_t::cbegin;
            using base_t::cend;
            using base_t::empty;
            using base_t::end;
            using base_t::size;

            using const_iterator = typename base_t::const_iterator;
            using iterator       = typename base_t::iterator;
            using value_type     = typename base_t::value_type;

            /// Value stored at key, inserted at its sorted position if missing.
            T& operator[](TCoord key)
            {
                base_t& data = *this;
                if (data.empty())
                {
                    data.emplace_back(key, T{});
                    m_hint = 0;
                    return data[0].second;
                }
                std::size_t i = m_hint;
                if (key > data[i].first)
                {
                    while (i + 1 < data.size() && key > data[i + 1].first)
                    {
                        ++i;
                    }
                    // key is in (data[i].first, data[i+1].first] or beyond the end
                    m_hint = i + 1;
                    if (i + 1 < data.size() && key == data[i + 1].first)
                    {
                        return data[i + 1].second;
                    }
                    return data.emplace(data.begin() + static_cast<std::ptrdiff_t>(i + 1), key, T{})->second;
                }
                while (i > 0 && key < data[i].first)
                {
                    --i;
                }
                if (key == data[i].first)
                {
                    m_hint = i;
                    return data[i].second;
                }
                // key is before data[0] (i == 0) or between data[i] and data[i+1]
                const std::size_t pos = key < data[i].first ? i : i + 1;
                m_hint                = pos;
                return data.emplace(data.begin() + static_cast<std::ptrdiff_t>(pos), key, T{})->second;
            }

            void clear()
            {
                base_t::clear();
                m_hint = 0;
            }

          private:

            /// Index of the last row touched.
            std::size_t m_hint = 0;
        };

        // Storage of a LevelCellList: sorted vectors by default, std::map and
        // forward lists with SAMURAI_CELL_LIST_CONTAINER=list_of_intervals.
#ifdef SAMURAI_CELL_LIST_CONTAINER_LIST_OF_INTERVALS
        template <typename TCoord, typename T>
        using cell_list_map_t = std::map<TCoord, T>;

        template <typename TCoord, typename TIndex>
        using cell_list_intervals_t = ListOfIntervals<TCoord, TIndex>;
#else
        template <typename TCoord, typename T>
        using cell_list_map_t = FlatMap<TCoord, T>;

        template <typename TCoord, typename TIndex>
        using cell_list_intervals_t = FlatListOfIntervals<TCoord, TIndex>;
#endif

        /// Type helper to create nested maps with final interval list
        template <typename TCoord, typename TIntervalList, std::size_t N>
        struct PartialGrid
        {
            using next_type = typename PartialGrid<TCoord, TIntervalList, N - 1>::type;
            using type      = cell_list_map_t<TCoord, next_type>;
        };

        template <typename TCoord, typename TIntervalList>
        struct PartialGrid<TCoord, TIntervalList, 0>
        {
            using type = TIntervalList;
        };

        /** Nested map accessor.
         *
         * It is separated from the LevelCellList in order to automatically
         * manage the constness of the context (avoid duplicated code).
         */
        template <typename GridYZ, typename Index>
        SAMURAI_INLINE decltype(auto) access_grid_yz(GridYZ& grid_yz, const Index&, std::integral_constant<std::size_t, 0>)
        {
            // For the first dimension, we return the interval list
            return grid_yz;
        }

        template <typename GridYZ, typename Index, std::size_t dim>
        SAMURAI_INLINE decltype(auto) access_grid_yz(GridYZ& grid_yz, const Index& index, std::integral_constant<std::size_t, dim>)
        {
            // For other dimensions, we dive into the nested map
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
        using list_interval_t     = detail::cell_list_intervals_t<coord_index_t, index_t>;
        using coords_t            = xt::xtensor_fixed<double, xt::xshape<dim>>;

        /// Sparse dim-1 array that points to the interval lists along the x
        /// axis.
        using grid_t = typename detail::PartialGrid<coord_index_t, list_interval_t, dim - 1>::type;

        LevelCellList();
        LevelCellList(std::size_t level);
        LevelCellList(std::size_t level, const coords_t& origin_point, double scaling_factor);

        const list_interval_t& operator[](const index_yz_t& index) const;
        list_interval_t& operator[](const index_yz_t& index);

        const grid_t& grid_yz() const;

        std::size_t level() const;

        bool empty() const;

        void to_stream(std::ostream& os) const;

        void add_cell(const Cell<dim, interval_t>& cell);

        auto& origin_point() const;
        double scaling_factor() const;

        void clear();

      private:

        grid_t m_grid_yz; ///< Sparse dim-1 array that points to the interval
                          ///< lists along the x axis.
        std::size_t m_level;
        coords_t m_origin_point;
        double m_scaling_factor = 1;
    };

    //////////////////////////////////
    // LevelCellList implementation //
    //////////////////////////////////
    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE LevelCellList<Dim, TInterval>::LevelCellList()
        : m_level{0}
    {
        m_origin_point.fill(0);
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE LevelCellList<Dim, TInterval>::LevelCellList(std::size_t level)
        : m_level{level}
    {
        m_origin_point.fill(0);
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE LevelCellList<Dim, TInterval>::LevelCellList(std::size_t level, const coords_t& origin_point, double scaling_factor)
        : m_level{level}
        , m_origin_point(origin_point)
        , m_scaling_factor(scaling_factor)
    {
    }

    /// Constant access to the interval list at given dim-1 coordinates
    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE auto LevelCellList<Dim, TInterval>::operator[](const index_yz_t& index) const -> const list_interval_t&
    {
        return detail::access_grid_yz(m_grid_yz, index, std::integral_constant<std::size_t, dim - 1>{});
    }

    /// Mutable access to the interval list at given dim-1 coordinates
    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE auto LevelCellList<Dim, TInterval>::operator[](const index_yz_t& index) -> list_interval_t&
    {
        return detail::access_grid_yz(m_grid_yz, index, std::integral_constant<std::size_t, dim - 1>{});
    }

    /// Underlying sparse array
    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE auto LevelCellList<Dim, TInterval>::grid_yz() const -> const grid_t&
    {
        return m_grid_yz;
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE std::size_t LevelCellList<Dim, TInterval>::level() const
    {
        return m_level;
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE bool LevelCellList<Dim, TInterval>::empty() const
    {
        return m_grid_yz.empty();
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE void LevelCellList<Dim, TInterval>::to_stream(std::ostream& os) const
    {
        os << "LevelCellList\n";
        os << "=============\n";
        os << "TODO\n";
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE void LevelCellList<Dim, TInterval>::add_cell(const Cell<dim, interval_t>& cell)
    {
        using namespace xt::placeholders;

        (*this)[xt::view(cell.indices, xt::range(1, _))].add_point(cell.indices[0]);
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE auto& LevelCellList<Dim, TInterval>::origin_point() const
    {
        return m_origin_point;
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE double LevelCellList<Dim, TInterval>::scaling_factor() const
    {
        return m_scaling_factor;
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE void LevelCellList<Dim, TInterval>::clear()
    {
        m_grid_yz.clear();
    }

    template <std::size_t Dim, class TInterval>
    SAMURAI_INLINE std::ostream& operator<<(std::ostream& out, const LevelCellList<Dim, TInterval>& level_cell_list)
    {
        level_cell_list.to_stream(out);
        return out;
    }

} // namespace samurai
