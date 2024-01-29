// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>
#include <iterator>
#include <limits>
#include <vector>

#ifdef SAMURAI_WITH_MPI
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#endif

#include <fmt/color.h>
#include <fmt/format.h>

#include "algorithm.hpp"
#include "box.hpp"
#include "interval.hpp"
#include "level_cell_list.hpp"
#include "mesh_interval.hpp"
#include "samurai_config.hpp"
#include "subset/subset_op_base.hpp"
#include "utils.hpp"

using namespace xt::placeholders;

namespace samurai
{

    template <class LCA, bool is_const>
    class LevelCellArray_iterator;

    template <class iterator>
    class LevelCellArray_reverse_iterator : public std::reverse_iterator<iterator>
    {
      public:

        using base_type  = std::reverse_iterator<iterator>;
        using coord_type = typename iterator::coord_type;

        explicit LevelCellArray_reverse_iterator(iterator&& it)
            : base_type(std::move(it))
        {
        }

        const coord_type index() const
        {
            iterator it = this->base();
            return (--it).index();
        }

        std::size_t level() const
        {
            iterator it = this->base();
            return (--it).level();
        }
    };

    ///////////////////////////////
    // LevelCellArray definition //
    ///////////////////////////////
    template <std::size_t Dim, class TInterval = default_config::interval_t>
    class LevelCellArray
    {
      public:

        static constexpr auto dim = Dim;
        using interval_t          = TInterval;
        using index_t             = typename interval_t::index_t;
        using value_t             = typename interval_t::value_t;
        using mesh_interval_t     = MeshInterval<Dim, TInterval>;

        using iterator               = LevelCellArray_iterator<LevelCellArray<Dim, TInterval>, false>;
        using reverse_iterator       = LevelCellArray_reverse_iterator<iterator>;
        using const_iterator         = LevelCellArray_iterator<const LevelCellArray<Dim, TInterval>, true>;
        using const_reverse_iterator = LevelCellArray_reverse_iterator<const_iterator>;

        using index_type = std::array<value_t, dim>;

        LevelCellArray() = default;
        LevelCellArray(const LevelCellList<Dim, TInterval>& lcl);

        template <class F, class... CT>
        LevelCellArray(subset_operator<F, CT...> set);

        LevelCellArray(std::size_t level, const Box<value_t, dim>& box);
        LevelCellArray(std::size_t level, const Box<double, dim>& box);
        LevelCellArray(std::size_t level);

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        reverse_iterator rbegin();
        reverse_iterator rend();

        const_reverse_iterator rbegin() const;
        const_reverse_iterator rend() const;
        const_reverse_iterator rcbegin() const;
        const_reverse_iterator rcend() const;

        /// Display to the given stream
        void to_stream(std::ostream& os) const;

        template <typename... T>
        const interval_t& get_interval(const interval_t& interval, T... index) const;

        const interval_t& get_interval(const interval_t& interval, const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const;

        const interval_t& get_interval(const xt::xtensor_fixed<value_t, xt::xshape<dim>>& coord) const;

        template <typename... T>
        index_t get_index(value_t i, T... index) const;
        index_t get_index(value_t i, const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& others) const;

        void update_index();

        //// checks whether the container is empty
        bool empty() const;

        //// Gives the number of intervals in each dimension
        auto shape() const;

        //// Gives the total number of intervals
        auto nb_intervals() const;

        //// Gives the number of cells
        std::size_t nb_cells() const;

        const std::vector<interval_t>& operator[](std::size_t d) const;
        std::vector<interval_t>& operator[](std::size_t d);

        const std::vector<std::size_t>& offsets(std::size_t d) const;
        std::vector<std::size_t>& offsets(std::size_t d);

        std::size_t level() const;

        auto min_indices() const;
        auto max_indices() const;
        auto minmax_indices() const;

      private:

#ifdef SAMURAI_WITH_MPI
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive& ar, const unsigned long)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                ar& m_cells[d];
            }
            for (std::size_t d = 0; d < dim - 1; ++d)
            {
                ar& m_offsets[d];
            }
            ar& m_level;
        }
#endif

        /// Recursive construction from a level cell list along dimension > 0
        template <typename TGrid, std::size_t N>
        void init_from_level_cell_list(const TGrid& grid, std::array<value_t, dim - 1> index, std::integral_constant<std::size_t, N>);

        /// Recursive construction from a level cell list for the dimension 0
        template <typename TIntervalList>
        void init_from_level_cell_list(const TIntervalList& interval_list,
                                       const std::array<value_t, dim - 1>& index,
                                       std::integral_constant<std::size_t, 0>);

        void init_from_box(const Box<value_t, dim>& box);

        std::array<std::vector<interval_t>, dim> m_cells;        ///< All intervals in every direction
        std::array<std::vector<std::size_t>, dim - 1> m_offsets; ///< Offsets in interval list for each dim >
                                                                 ///< 1
        std::size_t m_level = 0;
    };

    ////////////////////////////////////////
    // LevelCellArray_iterator definition //
    ////////////////////////////////////////
    namespace detail
    {
        template <class LCA, bool is_const>
        struct LevelCellArray_iterator_types
        {
            using value_type          = typename LCA::interval_t;
            using index_type          = std::vector<value_type>;
            using index_type_iterator = std::conditional_t<is_const, typename index_type::const_iterator, typename index_type::iterator>;
            using const_index_type_iterator = typename index_type::const_iterator;
            using reference                 = typename index_type_iterator::reference;
            using pointer                   = typename index_type_iterator::pointer;
            using difference_type           = typename index_type_iterator::difference_type;
        };
    } // namespace detail

    template <class LCA, bool is_const>
    class LevelCellArray_iterator
        : public xtl::xrandom_access_iterator_base3<LevelCellArray_iterator<LCA, is_const>, detail::LevelCellArray_iterator_types<LCA, is_const>>
    {
      public:

        static constexpr std::size_t dim = LCA::dim;
        using self_type                  = LevelCellArray_iterator<LCA, is_const>;
        using iterator_type              = detail::LevelCellArray_iterator_types<LCA, is_const>;
        using value_type                 = typename iterator_type::value_type;
        using index_type                 = typename iterator_type::index_type;
        using index_type_iterator        = typename iterator_type::index_type_iterator;
        using const_index_type_iterator  = typename iterator_type::const_index_type_iterator;
        using iterator_container         = std::array<index_type_iterator, dim>;
        using reference                  = typename iterator_type::reference;
        using pointer                    = typename iterator_type::pointer;
        using difference_type            = typename iterator_type::difference_type;
        using iterator_category          = std::random_access_iterator_tag;

        using offset_type          = std::vector<std::size_t>;
        using offset_type_iterator = std::array<typename offset_type::const_iterator, dim - 1>;

        using coord_type = xt::xtensor_fixed<typename value_type::value_t, xt::xshape<dim - 1>>;

        LevelCellArray_iterator(LCA* lca, offset_type_iterator&& offset_index, iterator_container&& current_index, coord_type&& index);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;
        const coord_type& index() const;
        std::size_t level() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

      private:

        LCA* p_lca;
        offset_type_iterator m_offset_index;
        iterator_container m_current_index;
        mutable coord_type m_index;
    };

    ///////////////////////////////////
    // LevelCellArray implementation //
    ///////////////////////////////////
    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(const LevelCellList<Dim, TInterval>& lcl)
        : m_level(lcl.level())
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
        if (!lcl.empty())
        {
            // Filling cells and offsets from the level cell list
            init_from_level_cell_list(lcl.grid_yz(), {}, std::integral_constant<std::size_t, dim - 1>{});
            // Additionnal offset so that [m_offset[i], m_offset[i+1][ is always
            // valid.
            for (std::size_t d = 0; d < dim - 1; ++d)
            {
                m_offsets[d].emplace_back(m_cells[d].size());
            }
        }
    }

    template <std::size_t Dim, class TInterval>
    template <class F, class... CT>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(subset_operator<F, CT...> set)
    {
        LevelCellList<Dim, TInterval> lcl{set.level()};

        set(
            [&lcl](const auto& i, const auto& index)
            {
                lcl[index].add_interval(i);
            });
        *this = {lcl};
    }

    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(std::size_t level, const Box<value_t, dim>& box)
        : m_level{level}
    {
        init_from_box(box);
    }

    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(std::size_t level, const Box<double, dim>& box)
        : m_level(level)
    {
        using box_t   = Box<value_t, dim>;
        using point_t = typename box_t::point_t;

        point_t start_pt = box.min_corner() * static_cast<double>(1 << level);
        point_t end_pt   = box.max_corner() * static_cast<double>(1 << level);
        init_from_box(box_t{start_pt, end_pt});
    }

    template <std::size_t Dim, class TInterval>
    inline LevelCellArray<Dim, TInterval>::LevelCellArray(std::size_t level)
        : m_level{level}
    {
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::begin() -> iterator
    {
        typename iterator::offset_type_iterator offset_index;
        typename iterator::iterator_container current_index;
        typename iterator::coord_type index;

        for (std::size_t d = 0; d < dim; ++d)
        {
            current_index[d] = m_cells[d].begin();
        }

        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            offset_index[d] = m_offsets[d].cbegin();
            index[d]        = current_index[d + 1]->start;
        }
        return iterator(this, std::move(offset_index), std::move(current_index), std::move(index));
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::end() -> iterator
    {
        typename iterator::offset_type_iterator offset_index;
        typename iterator::iterator_container current_index;
        typename iterator::coord_type index;

        for (std::size_t d = 0; d < dim; ++d)
        {
            current_index[d] = m_cells[d].end() - 1;
        }
        ++current_index[0];

        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            offset_index[d] = m_offsets[d].cend() - 2;
            index[d]        = current_index[d + 1]->end - 1;
        }

        return iterator(this, std::move(offset_index), std::move(current_index), std::move(index));
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::cbegin() const -> const_iterator
    {
        typename const_iterator::offset_type_iterator offset_index;
        typename const_iterator::iterator_container current_index;
        typename const_iterator::coord_type index;

        for (std::size_t d = 0; d < dim; ++d)
        {
            current_index[d] = m_cells[d].cbegin();
        }

        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            offset_index[d] = m_offsets[d].cbegin();
            index[d]        = current_index[d + 1]->start;
        }
        return const_iterator(this, std::move(offset_index), std::move(current_index), std::move(index));
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::cend() const -> const_iterator
    {
        typename const_iterator::offset_type_iterator offset_index;
        typename const_iterator::iterator_container current_index;
        typename const_iterator::coord_type index;

        for (std::size_t d = 0; d < dim; ++d)
        {
            current_index[d] = m_cells[d].cend() - 1;
        }
        ++current_index[0];

        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            offset_index[d] = m_offsets[d].cend() - 2;
            index[d]        = current_index[d + 1]->end - 1;
        }

        return const_iterator(this, std::move(offset_index), std::move(current_index), std::move(index));
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::end() const -> const_iterator
    {
        return cend();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rend() -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rbegin() const -> const_reverse_iterator
    {
        return rcbegin();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rend() const -> const_reverse_iterator
    {
        return rcend();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rcbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::rcend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    /**
     * Return the x-interval satisfying the input parameters
     *
     * @param interval The desired x-interval.
     * @param index The desired indices for the other dimensions.
     */
    template <std::size_t Dim, class TInterval>
    template <typename... T>
    inline auto LevelCellArray<Dim, TInterval>::get_interval(const interval_t& interval, T... index) const -> const interval_t&
    {
        auto row = find(*this, {interval.start, index...});
        return m_cells[0][static_cast<std::size_t>(row)];
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::get_interval(const interval_t& interval,
                                                             const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const
        -> const interval_t&
    {
        xt::xtensor_fixed<value_t, xt::xshape<dim>> point;
        point[0]                         = interval.start;
        xt::view(point, xt::range(1, _)) = index;
        auto row                         = find(*this, point);
        return m_cells[0][static_cast<std::size_t>(row)];
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::get_interval(const xt::xtensor_fixed<value_t, xt::xshape<dim>>& coord) const
        -> const interval_t&
    {
        auto row = find(*this, coord);
        return m_cells[0][static_cast<std::size_t>(row)];
    }

    template <std::size_t Dim, class TInterval>
    template <typename... T>
    inline auto LevelCellArray<Dim, TInterval>::get_index(value_t i, T... index) const -> index_t
    {
        return get_interval({i, i + 1}, index...).index + i;
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::get_index(value_t i, const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& others) const
        -> index_t
    {
        return get_interval({i, i + 1}, others).index + i;
    }

    /**
     * Update the index in the x-intervals allowing to navigate in the
     * Field data structure.
     */
    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::update_index()
    {
        std::size_t acc_size = 0;
        for_each_interval(*this,
                          [&](auto, auto& interval, auto)
                          {
                              interval.index = safe_subs<index_t>(acc_size, interval.start);
                              acc_size += interval.size();
                          });
    }

    template <std::size_t Dim, class TInterval>
    inline bool LevelCellArray<Dim, TInterval>::empty() const
    {
        return m_cells[0].empty();
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::shape() const
    {
        std::array<std::size_t, dim> output;
        for (std::size_t d = 0; d < dim; ++d)
        {
            output[d] = m_cells[d].size();
        }
        return output;
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::nb_intervals() const
    {
        std::size_t s = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            s += m_cells[d].size();
        }
        return s;
    }

    template <std::size_t Dim, class TInterval>
    inline std::size_t LevelCellArray<Dim, TInterval>::nb_cells() const
    {
        auto op = [](std::size_t i, const auto& interval)
        {
            return i + interval.size();
        };

        return std::accumulate(m_cells[0].cbegin(), m_cells[0].cend(), std::size_t(0), op);
    }

    template <std::size_t Dim, class TInterval>
    inline std::size_t LevelCellArray<Dim, TInterval>::level() const
    {
        return m_level;
    }

    /**
     * Return the maximum value that can take the end of an interval for each
     * direction.
     */
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::max_indices() const
    {
        std::array<value_t, dim> max;
        for (std::size_t d = 0; d < dim; ++d)
        {
            max[d] = std::max_element(m_cells[d].begin(),
                                      m_cells[d].end(),
                                      [](const auto& a, const auto& b)
                                      {
                                          return (a.end < b.end);
                                      })
                         ->end;
        }
        return max;
    }

    /**
     * Return the minimum value that can take the start of an interval for each
     * direction.
     */
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::min_indices() const
    {
        std::array<value_t, dim> min;
        for (std::size_t d = 0; d < dim; ++d)
        {
            min[d] = std::min_element(m_cells[d].begin(),
                                      m_cells[d].end(),
                                      [](const auto& a, const auto& b)
                                      {
                                          return (a.start < b.start);
                                      })
                         ->start;
        }
        return min;
    }

    /**
     * Return the minimum value that can take the start and
     * the maximum value that can take the end of an interval
     * for each direction.
     */
    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::minmax_indices() const
    {
        std::array<std::pair<value_t, value_t>, dim> minmax;
        auto min = min_indices();
        auto max = max_indices();
        for (std::size_t d = 0; d < dim; ++d)
        {
            minmax[d].first  = min[d];
            minmax[d].second = max[d];
        }
        return minmax;
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::operator[](std::size_t d) const -> const std::vector<interval_t>&
    {
        return m_cells[d];
    }

    template <std::size_t Dim, class TInterval>
    inline auto LevelCellArray<Dim, TInterval>::operator[](std::size_t d) -> std::vector<interval_t>&
    {
        return m_cells[d];
    }

    template <std::size_t Dim, class TInterval>
    inline const std::vector<std::size_t>& LevelCellArray<Dim, TInterval>::offsets(std::size_t d) const
    {
        assert(d > 0);
        return m_offsets[d - 1];
    }

    template <std::size_t Dim, class TInterval>
    inline std::vector<std::size_t>& LevelCellArray<Dim, TInterval>::offsets(std::size_t d)
    {
        assert(d > 0);
        return m_offsets[d - 1];
    }

    template <std::size_t Dim, class TInterval>
    template <typename TGrid, std::size_t N>
    inline void LevelCellArray<Dim, TInterval>::init_from_level_cell_list(const TGrid& grid,
                                                                          std::array<value_t, dim - 1> index,
                                                                          std::integral_constant<std::size_t, N>)
    {
        // Working interval
        interval_t curr_interval(0, 0, 0);

        // For each position along the Nth dimension
        for (const auto& point : grid)
        {
            // Coordinate along the Nth dimension
            const auto i = point.first;

            // Recursive call on the current position for the (N-1)th dimension
            index[N - 1]                      = i;
            const std::size_t previous_offset = m_cells[N - 1].size();
            init_from_level_cell_list(point.second, index, std::integral_constant<std::size_t, N - 1>{});

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
                    m_cells[N].emplace_back(curr_interval);

                    // ... and creating a new one.
                    curr_interval = interval_t(i, i + 1, static_cast<index_t>(m_offsets[N - 1].size()) - i);
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
                curr_interval = interval_t(i, i + 1, static_cast<index_t>(m_offsets[N - 1].size()) - i);
            }

            // Updating m_offsets (at each iteration since we are always
            // updating an interval)
            m_offsets[N - 1].emplace_back(previous_offset);
        }

        // Adding the working interval if valid
        if (curr_interval.is_valid())
        {
            m_cells[N].emplace_back(curr_interval);
        }
    }

    template <std::size_t Dim, class TInterval>
    template <typename TIntervalList>
    inline void LevelCellArray<Dim, TInterval>::init_from_level_cell_list(const TIntervalList& interval_list,
                                                                          const std::array<value_t, dim - 1>& /* index */,
                                                                          std::integral_constant<std::size_t, 0>)
    {
        // Along the X axis, simply copy the intervals in cells[0]
        std::copy(interval_list.begin(), interval_list.end(), std::back_inserter(m_cells[0]));
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::init_from_box(const Box<value_t, dim>& box)
    {
        auto dimensions = xt::cast<std::size_t>(box.length());
        auto start_pt   = box.min_corner();
        auto end_pt     = box.max_corner();

        std::size_t size = 1;
        for (std::size_t d = dim - 1; d > 0; --d)
        {
            m_offsets[d - 1].resize((dimensions[d] * size) + 1);
            for (std::size_t i = 0; i < (dimensions[d] * size) + 1; ++i)
            {
                m_offsets[d - 1][i] = i;
            }
            m_cells[d].resize(size);
            for (std::size_t i = 0; i < size; ++i)
            {
                m_cells[d][i] = {start_pt[d], end_pt[d], static_cast<index_t>(m_offsets[d - 1][i * dimensions[d]]) - start_pt[d]};
            }
            size *= dimensions[d];
        }

        m_cells[0].resize(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            m_cells[0][i] = {start_pt[0], end_pt[0], static_cast<index_t>(i * dimensions[0]) - start_pt[0]};
        }
    }

    template <std::size_t Dim, class TInterval>
    inline void LevelCellArray<Dim, TInterval>::to_stream(std::ostream& os) const
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            os << fmt::format(disable_color ? fmt::text_style() : fmt::emphasis::bold, "{:>10}", fmt::format("dim {}", d)) << std::endl;

            os << fmt::format("{:>20}", "cells = ");
            for (std::size_t ic = 0; ic < m_cells[d].size(); ++ic)
            {
                os << fmt::format(disable_color ? fmt::text_style() : fmt::emphasis::bold, "{}->", ic);
                os << m_cells[d][ic] << " ";
            }
            os << "\n" << std::endl;

            if (d > 0)
            {
                os << fmt::format("{:>20}", "offsets = ");
                os << fmt::format("{}\n", fmt::join(m_offsets[d - 1], " ")) << std::endl;
            }
        }
    }

    template <std::size_t Dim, class TInterval>
    inline bool operator==(const LevelCellArray<Dim, TInterval>& lca_1, const LevelCellArray<Dim, TInterval>& lca_2)
    {
        if (lca_1.level() != lca_2.level())
        {
            return false;
        }

        if (lca_1.shape() != lca_2.shape())
        {
            return false;
        }

        for (std::size_t i = 0; i < Dim; ++i)
        {
            if (lca_1[i] != lca_2[i])
            {
                return false;
            }
        }

        for (std::size_t i = 1; i < Dim; ++i)
        {
            if (lca_1.offsets(i) != lca_2.offsets(i))
            {
                return false;
            }
        }
        return true;
    }

    template <std::size_t Dim, class TInterval>
    inline std::ostream& operator<<(std::ostream& out, const LevelCellArray<Dim, TInterval>& level_cell_array)
    {
        level_cell_array.to_stream(out);
        return out;
    }

    ////////////////////////////////////////////
    // LevelCellArray_iterator implementation //
    ////////////////////////////////////////////

    template <class LCA, bool is_const>
    inline LevelCellArray_iterator<LCA, is_const>::LevelCellArray_iterator(LCA* lca,
                                                                           offset_type_iterator&& offset_index,
                                                                           iterator_container&& current_index,
                                                                           coord_type&& index)
        : p_lca(lca)
        , m_offset_index(std::move(offset_index))
        , m_current_index(std::move(current_index))
        , m_index(std::move(index))
    {
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator++() -> self_type&
    {
        if (m_current_index[0] == (*p_lca)[0].end())
        {
            return *this;
        }
        ++m_current_index[0];

        for (std::size_t d = 0; d < m_current_index.size() - 1; ++d)
        {
            auto dst = static_cast<std::size_t>(
                std::distance((*p_lca)[d].cbegin(), static_cast<const_index_type_iterator>(m_current_index[d])));
            if (dst == *(m_offset_index[d] + 1))
            {
                ++m_offset_index[d];
                ++m_index[d];
                if (m_index[d] == m_current_index[d + 1]->end)
                {
                    ++m_current_index[d + 1];
                    if (m_current_index[d + 1] != (*p_lca)[d + 1].end())
                    {
                        m_index[d] = m_current_index[d + 1]->start;
                    }
                }
            }
            else
            {
                break;
            }
        }
        return *this;
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator--() -> self_type&
    {
        if (m_current_index[0] == (*p_lca)[0].begin())
        {
            --m_current_index[0];
            return *this;
        }
        --m_current_index[0];

        for (std::size_t d = 0; d < m_current_index.size() - 1; ++d)
        {
            auto dst = static_cast<std::size_t>(
                std::distance((*p_lca)[d].cbegin(), static_cast<const_index_type_iterator>(m_current_index[d])));
            if (dst == *m_offset_index[d] - 1)
            {
                --m_offset_index[d];
                if (m_index[d] == m_current_index[d + 1]->start)
                {
                    if (m_current_index[d + 1] != (*p_lca)[d + 1].begin())
                    {
                        --m_current_index[d + 1];
                        m_index[d] = m_current_index[d + 1]->end - 1;
                    }
                }
                else
                {
                    --m_index[d];
                }
            }
            else
            {
                break;
            }
        }
        return *this;
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator+=(difference_type n) -> self_type&
    {
        for (difference_type i = 0; i < n; ++i)
        {
            ++(*this);
        }
        return *this;
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator-=(difference_type n) -> self_type&
    {
        for (difference_type i = 0; i < n; ++i)
        {
            --(*this);
        }
        return *this;
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_current_index[0] - rhs.m_current_index[0];
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator*() const -> reference
    {
        return *(m_current_index[0]);
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::operator->() const -> pointer
    {
        return std::addressof(this->operator*());
    }

    template <class LCA, bool is_const>
    inline auto LevelCellArray_iterator<LCA, is_const>::index() const -> const coord_type&
    {
        return m_index;
    }

    template <class LCA, bool is_const>
    inline std::size_t LevelCellArray_iterator<LCA, is_const>::level() const
    {
        return p_lca->level();
    }

    template <class LCA, bool is_const>
    inline bool LevelCellArray_iterator<LCA, is_const>::equal(const self_type& rhs) const
    {
        return p_lca == rhs.p_lca && m_current_index[0] == rhs.m_current_index[0];
    }

    template <class LCA, bool is_const>
    inline bool LevelCellArray_iterator<LCA, is_const>::less_than(const self_type& rhs) const
    {
        return p_lca == rhs.p_lca && m_current_index[0] < rhs.m_current_index[0];
    }

    template <class LCA, bool is_const>
    inline bool operator==(const LevelCellArray_iterator<LCA, is_const>& it1, const LevelCellArray_iterator<LCA, is_const>& it2)
    {
        return it1.equal(it2);
    }

    template <class LCA, bool is_const>
    inline bool operator<(const LevelCellArray_iterator<LCA, is_const>& it1, const LevelCellArray_iterator<LCA, is_const>& it2)
    {
        return it1.less_than(it2);
    }

    template <class LCA, bool is_const>
    inline bool operator==(const std::reverse_iterator<LevelCellArray_iterator<LCA, is_const>>& it1,
                           const std::reverse_iterator<LevelCellArray_iterator<LCA, is_const>>& it2)
    {
        return it1.base().equal(it2.base());
    }
} // namespace samurai
