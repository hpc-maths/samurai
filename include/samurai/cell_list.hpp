// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>

#include <fmt/color.h>

#include "level_cell_list.hpp"
#include "samurai_config.hpp"

namespace samurai
{

    /////////////////////////
    // CellList definition //
    /////////////////////////

    template <std::size_t dim_, class TInterval = default_config::interval_t, std::size_t max_size_ = default_config::max_level>
    class CellList
    {
      public:

        static constexpr auto dim      = dim_;
        static constexpr auto max_size = max_size_;

        using lcl_type = LevelCellList<dim, TInterval>;
        using coords_t = typename lcl_type::coords_t;

        CellList();
        CellList(const coords_t& origin_point, double scaling_factor);

        const lcl_type& operator[](std::size_t i) const;
        lcl_type& operator[](std::size_t i);

        void to_stream(std::ostream& os) const;

        auto& origin_point() const;
        auto scaling_factor() const;

        void clear();

      private:

        std::array<lcl_type, max_size + 1> m_cells;
    };

    /////////////////////////////
    // CellList implementation //
    /////////////////////////////

    /**
     * Default contructor which sets the level for each LevelCellArray.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline CellList<dim_, TInterval, max_size_>::CellList()
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            m_cells[level] = {level};
        }
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline CellList<dim_, TInterval, max_size_>::CellList(const coords_t& origin_point, double scaling_factor)
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            m_cells[level] = {level, origin_point, scaling_factor};
        }
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellList<dim_, TInterval, max_size_>::operator[](std::size_t i) const -> const lcl_type&
    {
        return m_cells[i];
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellList<dim_, TInterval, max_size_>::operator[](std::size_t i) -> lcl_type&
    {
        return m_cells[i];
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline void CellList<dim_, TInterval, max_size_>::to_stream(std::ostream& os) const
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            os << fmt::format(fg(fmt::color::crimson) | fmt::emphasis::bold, "Level {}\n", level);
            m_cells[level].to_stream(os);
            os << "\n";
        }
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto& CellList<dim_, TInterval, max_size_>::origin_point() const
    {
        return m_cells[0].origin_point();
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline auto CellList<dim_, TInterval, max_size_>::scaling_factor() const
    {
        return m_cells[0].scaling_factor();
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline void CellList<dim_, TInterval, max_size_>::clear()
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            m_cells[level].clear();
        }
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    inline std::ostream& operator<<(std::ostream& out, const CellList<dim_, TInterval, max_size_>& cell_list)
    {
        cell_list.to_stream(out);
        return out;
    }
} // namespace samurai
