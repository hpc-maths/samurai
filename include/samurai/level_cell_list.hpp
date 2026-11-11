// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <iostream>
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
        /// Sorted vector of (coordinate, value) pairs with a hinted operator[]:
        /// the rows of a level are visited in order, so the lookup starts from
        /// the last row touched instead of a tree search, and a new row is
        /// almost always appended.
        template <typename TCoord, typename T>
        class FlatMap
        {
          public:

            using value_type     = std::pair<TCoord, T>;
            using storage_t      = std::vector<value_type>;
            using iterator       = typename storage_t::iterator;
            using const_iterator = typename storage_t::const_iterator;

            T& operator[](TCoord key)
            {
                if (m_data.empty())
                {
                    m_data.emplace_back(key, T{});
                    m_hint = 0;
                    return m_data[0].second;
                }
                std::size_t i = m_hint;
                if (key > m_data[i].first)
                {
                    while (i + 1 < m_data.size() && key > m_data[i + 1].first)
                    {
                        ++i;
                    }
                    // key is in (m_data[i].first, m_data[i+1].first) or beyond the end
                    if (i + 1 < m_data.size() && key == m_data[i + 1].first)
                    {
                        m_hint = i + 1;
                        return m_data[i + 1].second;
                    }
                    m_hint = i + 1;
                    return m_data.emplace(m_data.begin() + static_cast<std::ptrdiff_t>(i + 1), key, T{})->second;
                }
                while (i > 0 && key < m_data[i].first)
                {
                    --i;
                }
                if (key == m_data[i].first)
                {
                    m_hint = i;
                    return m_data[i].second;
                }
                // either key < m_data[0].first (i == 0), or m_data[i].first < key < m_data[i+1].first
                const std::size_t pos = key < m_data[i].first ? i : i + 1;
                m_hint                = pos;
                return m_data.emplace(m_data.begin() + static_cast<std::ptrdiff_t>(pos), key, T{})->second;
            }

            bool empty() const
            {
                return m_data.empty();
            }

            std::size_t size() const
            {
                return m_data.size();
            }

            void clear()
            {
                m_data.clear();
                m_hint = 0;
            }

            iterator begin()
            {
                return m_data.begin();
            }

            iterator end()
            {
                return m_data.end();
            }

            const_iterator begin() const
            {
                return m_data.begin();
            }

            const_iterator end() const
            {
                return m_data.end();
            }

          private:

            storage_t m_data;
            std::size_t m_hint = 0;
        };

        /// Type helper to create nested flat maps with final interval list
        template <typename TCoord, typename TIntervalList, std::size_t N>
        struct PartialGrid
        {
            using next_type = typename PartialGrid<TCoord, TIntervalList, N - 1>::type;
            using type      = FlatMap<TCoord, next_type>;
        };

        template <typename TCoord, typename TIntervalList>
        struct PartialGrid<TCoord, TIntervalList, 0>
        {
            using type = TIntervalList;
        };

        /** Nested FlatMap accessor.
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
            // For other dimensions, we dive into the nested FlatMap
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
        using list_interval_t     = FlatListOfIntervals<coord_index_t, index_t>;
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
