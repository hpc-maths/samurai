#pragma once

#include <array>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "mure/box.hpp"
#include "mure/cell.hpp"
#include "mure/interval.hpp"
#include "mure/level_cell_list.hpp"
#include "mure/math.hpp"

namespace mure
{

    template<std::size_t Dim, class TInterval = Interval<int>>
    class LevelCellArray {
      public:
        constexpr static auto dim = Dim;
        using interval_t = TInterval;
        using index_t = typename interval_t::index_t;
        using coord_index_t = typename interval_t::value_t;

        LevelCellArray(LevelCellArray &&) = default;
        LevelCellArray &operator=(LevelCellArray &&) = default;

        LevelCellArray(const LevelCellArray &) = default;
        LevelCellArray &operator=(const LevelCellArray &) = default;

        /// Construct from a level cell list
        inline LevelCellArray(const LevelCellList<Dim, TInterval> &lcl = {});

        inline LevelCellArray(std::size_t level, Box<coord_index_t, dim> box);

        /// Display to the given stream
        void to_stream(std::ostream &out) const;

        /// Apply a given function to each interval along x
        template<typename TFunction>
        void for_each_interval_in_x(TFunction &&f) const;

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

        auto const &operator[](std::size_t d) const;
        auto &operator[](std::size_t d);

        //// Return a const reference to the offsets array
        //// for dimension d
        auto const &offsets(std::size_t d) const;

        //// Return a reference to the offsets array
        //// for dimension d
        auto &offsets(std::size_t d);

        //// Apply a function on each cell
        template<class TFunction>
        void for_each_cell(std::size_t level, TFunction &&func) const;

        //// Find the interval in x for the indices (ix, iy, iz)
        //// return -1 if not found
        inline index_t
        find(xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord) const;

        inline std::size_t find_on_dim(std::size_t d, std::size_t start_index,
                                       std::size_t end_index,
                                       coord_index_t coord) const
        {
            for (std::size_t i = start_index; i < end_index; ++i)
            {
                if (m_cells[d][i].contains(coord))
                {
                    return i;
                }
            }
            return std::numeric_limits<std::size_t>::max();
        }

        std::size_t get_level() const
        {
            return m_level;
        }

      private:
        /// Recursive construction from a level cell list along dimension > 0
        template<typename TGrid, std::size_t N>
        inline void initFromLevelCellList(
            TGrid const &grid,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> index,
            std::integral_constant<std::size_t, N>);

        /// Recursive construction from a level cell list for the dimension 0
        template<typename TIntervalList>
        inline void initFromLevelCellList(
            TIntervalList const &interval_list,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const &index,
            std::integral_constant<std::size_t, 0>);

        /// Recursive apply of a function on each interval along x, for
        /// dimension > 0
        template<typename TFunction, std::size_t N>
        inline void for_each_interval_in_x_impl(
            TFunction &&f,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> index,
            std::size_t start_index, std::size_t end_index,
            std::integral_constant<std::size_t, N>) const;

        /// Recursive apply of a function on each interval along x, for the
        /// dimension 0
        template<typename TFunction>
        inline void for_each_interval_in_x_impl(
            TFunction &&f,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const &index,
            std::size_t start_index, std::size_t end_index,
            std::integral_constant<std::size_t, 0>) const;

        /// Recursive apply of a function on each cell, for the dimension > 0
        template<typename TFunction, std::size_t N>
        inline void for_each_cell_impl(
            TFunction &&func,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
            std::size_t start_index, std::size_t end_index,
            std::integral_constant<std::size_t, N>) const;

        /// Recursive apply of a function on each cell, for the dimension 0
        template<typename TFunction>
        inline void for_each_cell_impl(
            TFunction &&func,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
            std::size_t start_index, std::size_t end_index,
            std::integral_constant<std::size_t, 0>) const;

        template<std::size_t N>
        inline index_t
        find_impl(std::size_t start_index, std::size_t end_index,
                  xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord,
                  std::integral_constant<std::size_t, N>) const;

        inline index_t
        find_impl(std::size_t start_index, std::size_t end_index,
                  xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord,
                  std::integral_constant<std::size_t, 0>) const;

      private:
        std::array<std::vector<interval_t>, dim>
            m_cells; ///< All intervals in every direction
        std::array<std::vector<std::size_t>, dim - 1>
            m_offsets; ///< Offsets in interval list for each dim > 1
        std::size_t m_level;
    };

    template<std::size_t Dim, class TInterval>
    LevelCellArray<Dim, TInterval>::LevelCellArray(
        LevelCellList<Dim, TInterval> const &lcl)
    {
        /* Estimating reservation size
         *
         * NOTE: the estimation takes time, more than the time needed for
         * reallocating the vectors... Maybe 2 other solutions:
         * - (highly) overestimating the needed size since the memory will be
         * actually allocated only when touched (at least under Linux)
         * - cnt_x and cnt_yz updated in LevelCellList during the filling
         * process
         *
         * NOTE2: in fact, hard setting the optimal values for cnt_x and cnt_yz
         * doesn't speedup things, strang...
         */

        // Filling cells and offsets from the level cell list
        initFromLevelCellList(lcl.grid_yz(), {},
                              std::integral_constant<std::size_t, dim - 1>{});
        m_level = lcl.get_level();
        // Additionnal offset so that [m_offset[i], m_offset[i+1][ is always
        // valid.
        for (std::size_t N = 0; N < dim - 1; ++N)
            m_offsets[N].push_back(m_cells[N].size());
    }

    template<std::size_t Dim, class TInterval>
    LevelCellArray<Dim, TInterval>::LevelCellArray(std::size_t level,
                                                   Box<coord_index_t, dim> box)
        : m_level{level}
    {
        auto dimensions =
            static_cast<xt::xtensor_fixed<std::size_t, xt::xshape<dim>>>(
                box.length());
        auto start = box.min_corner();
        auto end = box.max_corner();

        std::size_t size = 1;
        for (std::size_t d = dim - 1; d > 0; --d)
        {
            m_offsets[d - 1].resize((dimensions[d] * size) + 1);
            for (std::size_t i = 0; i < (dimensions[d] * size) + 1; ++i)
                m_offsets[d - 1][i] = i;
            m_cells[d].resize(size);
            for (std::size_t i = 0; i < size; ++i)
                m_cells[d][i] = {
                    start[d], end[d],
                    static_cast<index_t>(m_offsets[d - 1][i * dimensions[d]]) -
                        start[d]};
            size *= dimensions[d];
        }

        m_cells[0].resize(size);
        for (std::size_t i = 0; i < size; ++i)
            // m_cells[0][i] = {start[0], end[0], 0};
            m_cells[0][i] = {start[0], end[0],
                             static_cast<index_t>(i * dimensions[0]) -
                                 start[0]};
    }

    template<std::size_t Dim, class TInterval>
    inline bool LevelCellArray<Dim, TInterval>::empty() const
    {
        return m_cells[0].empty();
    }

    template<std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::shape() const
    {
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> output;
        for (std::size_t i = 0; i < dim; ++i)
        {
            output[i] = m_cells[i].size();
        }
        return output;
    }

    template<std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::nb_cells() const
    {
        auto op = [](auto &&init, auto const &interval) {
            return std::move(init) + interval.size();
        };
        return std::accumulate(m_cells[0].cbegin(), m_cells[0].cend(),
                               std::size_t(0), op);
    }

    template<std::size_t Dim, class TInterval>
    auto const &LevelCellArray<Dim, TInterval>::operator[](std::size_t d) const
    {
        return m_cells[d];
    }

    template<std::size_t Dim, class TInterval>
    auto &LevelCellArray<Dim, TInterval>::operator[](std::size_t d)
    {
        return m_cells[d];
    }

    template<std::size_t Dim, class TInterval>
    auto const &LevelCellArray<Dim, TInterval>::offsets(std::size_t d) const
    {
        assert(d > 0);
        return m_offsets[d - 1];
    }

    template<std::size_t Dim, class TInterval>
    auto &LevelCellArray<Dim, TInterval>::offsets(std::size_t d)
    {
        assert(d > 0);
        return m_offsets[d - 1];
    }

    template<std::size_t Dim, class TInterval>
    template<typename TGrid, std::size_t N>
    void LevelCellArray<Dim, TInterval>::initFromLevelCellList(
        TGrid const &grid,
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> index,
        std::integral_constant<std::size_t, N>)
    {
        // Working interval
        interval_t curr_interval(0, 0, 0);

        // For each position along the Nth dimension
        for (auto const &point : grid)
        {
            // Coordinate along the Nth dimension
            const auto i = point.first;

            // Recursive call on the current position for the (N-1)th dimension
            index[N - 1] = i;
            const std::size_t previous_offset = m_cells[N - 1].size();
            initFromLevelCellList(point.second, index,
                                  std::integral_constant<std::size_t, N - 1>{});

            /* Since we move on a sparse storage, each coordinate have non-empty
             * co-dimensions So the question is, are we continuing an existing
             * interval or have we jump to another one.
             *
             * WARNING: we are supposing that the sparse array of dimension
             * dim-1 has no empty entry. Otherwise, we should check that the
             * recursive call has do something by comparing previous_offset
             * with the size of m_cells[N-1].
             */
            if (curr_interval.is_valid())
            {
                // If the coordinate has jump out of the current interval
                if (i > curr_interval.end)
                {
                    // Adding the previous interval...
                    m_cells[N].push_back(curr_interval);

                    // ... and creating a new one.
                    curr_interval = interval_t(
                        i, i + 1,
                        static_cast<index_t>(m_offsets[N - 1].size()) - i);
                }
                else
                {
                    // Otherwise, we are just continuing the current interval
                    ++curr_interval.end;
                }
            }
            else
            {
                // If there is no current interval (at the beginning of the
                // loop) we create a new one.
                curr_interval = interval_t(
                    i, i + 1,
                    static_cast<index_t>(m_offsets[N - 1].size()) - i);
            }

            // Updating m_offsets (at each iteration since we are always
            // updating an interval)
            m_offsets[N - 1].push_back(previous_offset);
        }

        // Adding the working interval if valid
        if (curr_interval.is_valid())
            m_cells[N].push_back(curr_interval);
    }

    template<std::size_t Dim, class TInterval>
    template<typename TIntervalList>
    void LevelCellArray<Dim, TInterval>::initFromLevelCellList(
        TIntervalList const &interval_list,
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const
            & /* index */,
        std::integral_constant<std::size_t, 0>)
    {
        // Along the X axis, simply copy the intervals in cells[0]
        std::copy(interval_list.begin(), interval_list.end(),
                  std::back_inserter(m_cells[0]));
    }

    template<std::size_t Dim, class TInterval>
    void LevelCellArray<Dim, TInterval>::to_stream(std::ostream &out) const
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            out << "Dim " << d << std::endl;

            out << "\tcells = ";
            for (auto const &interval : m_cells[d])
                out << interval << " ";
            out << std::endl;

            if (d > 0)
            {
                out << "\toffsets = ";
                for (auto const &v : m_offsets[d - 1])
                    out << v << " ";
                out << std::endl;
            }
        }
    }

    template<std::size_t Dim, class TInterval>
    template<typename TFunction>
    void
    LevelCellArray<Dim, TInterval>::for_each_interval_in_x(TFunction &&f) const
    {
        for_each_interval_in_x_impl(
            std::forward<TFunction>(f), {}, 0, m_cells[dim - 1].size(),
            std::integral_constant<std::size_t, dim - 1>{});
    }

    template<std::size_t Dim, class TInterval>
    template<typename TFunction, std::size_t N>
    void LevelCellArray<Dim, TInterval>::for_each_interval_in_x_impl(
        TFunction &&f,
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> index,
        std::size_t start_index, std::size_t end_index,
        std::integral_constant<std::size_t, N>) const
    {
        for (std::size_t i = start_index; i < end_index; ++i)
        {
            auto const &interval = m_cells[N][i];
            for (coord_index_t c = interval.start; c < interval.end; ++c)
            {
                index[N - 1] = c;
                auto off_ind = static_cast<std::size_t>(interval.index + c);
                for_each_interval_in_x_impl(
                    std::forward<TFunction>(f), index,
                    m_offsets[N - 1][off_ind], m_offsets[N - 1][off_ind + 1],
                    std::integral_constant<std::size_t, N - 1>{});
            }
        }
    }

    template<std::size_t Dim, class TInterval>
    template<typename TFunction>
    void LevelCellArray<Dim, TInterval>::for_each_interval_in_x_impl(
        TFunction &&f,
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const &index,
        std::size_t start_index, std::size_t end_index,
        std::integral_constant<std::size_t, 0>) const
    {
        for (std::size_t i = start_index; i < end_index; ++i)
        {
            std::forward<TFunction>(f)(index, m_cells[0][i]);
        }
    }

    template<std::size_t Dim, class TInterval>
    template<typename TFunction>
    void LevelCellArray<Dim, TInterval>::for_each_cell(std::size_t level,
                                                       TFunction &&f) const
    {
        for_each_cell_impl(std::forward<TFunction>(f), {}, 0,
                           m_cells[dim - 1].size(),
                           std::integral_constant<std::size_t, dim - 1>{});
    }

    template<std::size_t Dim, class TInterval>
    template<typename TFunction, std::size_t N>
    void LevelCellArray<Dim, TInterval>::for_each_cell_impl(
        TFunction &&f, xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
        std::size_t start_index, std::size_t end_index,
        std::integral_constant<std::size_t, N>) const
    {
        for (std::size_t i = start_index; i < end_index; ++i)
        {
            auto const &interval = m_cells[N][i];
            for (coord_index_t c = interval.start; c < interval.end; ++c)
            {
                index[N] = c;
                auto off_ind = static_cast<std::size_t>(interval.index + c);
                for_each_cell_impl(
                    std::forward<TFunction>(f), index,
                    m_offsets[N - 1][off_ind], m_offsets[N - 1][off_ind + 1],
                    std::integral_constant<std::size_t, N - 1>{});
            }
        }
    }

    template<std::size_t Dim, class TInterval>
    template<typename TFunction>
    void LevelCellArray<Dim, TInterval>::for_each_cell_impl(
        TFunction &&f, xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
        std::size_t start_index, std::size_t end_index,
        std::integral_constant<std::size_t, 0>) const
    {
        for (std::size_t i = start_index; i < end_index; ++i)
        {
            auto const &interval = m_cells[0][i];
            for (coord_index_t c = interval.start; c < interval.end; ++c)
            {
                index[0] = c;
                Cell<coord_index_t, dim> cell{
                    m_level, index,
                    static_cast<std::size_t>(interval.index + c)};
                std::forward<TFunction>(f)(cell);
            }
        }
    }

    template<std::size_t Dim, class TInterval>
    typename LevelCellArray<Dim, TInterval>::index_t
    LevelCellArray<Dim, TInterval>::find(
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord) const
    {
        return find_impl(0, m_cells[dim - 1].size(), coord,
                         std::integral_constant<std::size_t, dim - 1>{});
    }

    template<std::size_t Dim, class TInterval>
    template<std::size_t N>
    typename LevelCellArray<Dim, TInterval>::index_t
    LevelCellArray<Dim, TInterval>::find_impl(
        std::size_t start_index, std::size_t end_index,
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord,
        std::integral_constant<std::size_t, N>) const
    {
        index_t find_index = -1;
        for (std::size_t i = start_index; i < end_index; ++i)
        {
            auto const &interval = m_cells[N][i];
            if (interval.contains(coord[N]))
            {
                auto off_ind =
                    static_cast<std::size_t>(interval.index + coord[N]);
                find_index = find_impl(
                    m_offsets[N - 1][off_ind], m_offsets[N - 1][off_ind + 1],
                    coord, std::integral_constant<std::size_t, N - 1>{});
            }
        }
        return find_index;
    }

    template<std::size_t Dim, class TInterval>
    typename LevelCellArray<Dim, TInterval>::index_t
    LevelCellArray<Dim, TInterval>::find_impl(
        std::size_t start_index, std::size_t end_index,
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord,
        std::integral_constant<std::size_t, 0>) const
    {
        for (std::size_t i = start_index; i < end_index; ++i)
        {
            auto const &interval = m_cells[0][i];
            if (interval.contains(coord[0]))
            {
                return static_cast<index_t>(i);
            }
        }
        return -1;
    }

    template<std::size_t Dim, class TInterval>
    bool operator==(const LevelCellArray<Dim, TInterval> &lca_1,
                    const LevelCellArray<Dim, TInterval> &lca_2)
    {
        if (lca_1.get_level() != lca_2.get_level())
            return false;

        if (lca_1.shape() != lca_2.shape())
            return false;

        for (std::size_t i = 0; i < Dim; ++i)
        {
            if (lca_1[i] != lca_2[i])
                return false;
        }

        for (std::size_t i = 0; i < Dim - 1; ++i)
        {
            if (lca_1.offsets(i) != lca_2.offsets(i))
                return false;
        }
        return true;
    }

    template<std::size_t Dim, class TInterval>
    std::ostream &
    operator<<(std::ostream &out,
               LevelCellArray<Dim, TInterval> const &level_cell_array)
    {
        level_cell_array.to_stream(out);
        return out;
    }

} // namespace mure
