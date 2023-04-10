// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <algorithm>
#include <array>
#include <iterator>

// #include <xtensor/xadapt.hpp>
// #include <xtensor/xio.hpp>
// #include <spdlog/spdlog.h>
// #include <spdlog/fwd.h>

#include "../utils.hpp"
#include "node_op.hpp"

namespace samurai
{
    namespace detail
    {
        template <class T>
        inline T shift_value(T value, T shift)
        {
            return (shift >= 0) ? (value << shift) : (value >> (-shift));
        }
    } // namespace detail

    ////////////////////////////
    // subset_node definition //
    ////////////////////////////

    template <class T>
    class subset_node
    {
      public:

        using node_type                  = T;
        using mesh_type                  = typename node_type::mesh_type;
        static constexpr std::size_t dim = mesh_type::dim;
        using interval_t                 = typename mesh_type::interval_t;
        using coord_index_t              = typename interval_t::coord_index_t;
        using index_t                    = typename interval_t::index_t;

        subset_node(T&& node);

        void reset();
        bool eval(coord_index_t scan, std::size_t dim) const;
        void update(coord_index_t scan, coord_index_t sentinel);

        bool is_valid() const;
        bool is_empty() const;

        void decrement_dim(coord_index_t i);
        void increment_dim();

        coord_index_t min() const;
        coord_index_t max() const;
        std::size_t common_level() const;

        void set_shift(std::size_t ref_level, std::size_t common_level);

        const node_type& get_node() const;
        void get_interval_index(std::vector<std::size_t>& index) const;

      private:

        //! Shift between the ref_level and the node level
        int m_shift_ref = 0;
        //! Shift between the common_level and the node level
        int m_shift_common = 0;
        //! Minimum value between shift_ref and shift_common
        int m_shift = 0;
        //! The reference level where to compute the subset
        std::size_t m_ref_level = 0;
        //! The biggest level where all nodes of the subset can be compared
        std::size_t m_common_level = 0;
        //! The current dimension
        std::size_t m_d = 0;
        //! The position of the current interval for each dimension
        std::array<std::size_t, dim> m_index;
        //! The position inside the current interval for each dimension
        //! - ipos = 0 -> beginning
        //! - ipos = 1 -> end
        std::array<std::size_t, dim> m_ipos;
        //! The beginning of the portion of intervals to look for each dimension
        std::array<std::size_t, dim> m_start;
        //! The end of the portion of intervals to look for each dimension
        std::array<std::size_t, dim> m_end;
        std::array<std::size_t, dim> m_start_offset;
        std::array<std::size_t, dim> m_end_offset;
        std::array<coord_index_t, dim> m_current_value;
        std::array<std::vector<interval_t>, dim> m_work;
        std::array<std::vector<std::pair<std::size_t, std::size_t>>, dim> m_work_offsets;
        node_type m_node;
    };

    ////////////////////////////////
    // subset_node implementation //
    ////////////////////////////////

    template <class T>
    inline subset_node<T>::subset_node(T&& node)
        : m_node(std::forward<T>(node))
    {
    }

    /**
     * Check if the node is valid
     */
    template <class T>
    inline bool subset_node<T>::is_valid() const
    {
        return !(m_end[m_d] == m_start[m_d]);
    }

    /**
     * Check if the node has values
     */
    template <class T>
    inline bool subset_node<T>::is_empty() const
    {
        return m_node.is_empty();
    }

    /**
     * Reset the internal data structures to replay
     * the subset algorithm
     */
    template <class T>
    inline void subset_node<T>::reset()
    {
        m_d                 = dim - 1;
        m_start[m_d]        = 0;
        m_end[m_d]          = m_node.size(m_d);
        m_start_offset[m_d] = 0;
        m_end_offset[m_d]   = m_node.size(m_d);
        m_index[m_d]        = m_start[m_d];
        m_ipos[m_d]         = 0;

        if (m_start[m_d] != m_end[m_d])
        {
            m_current_value[m_d] = detail::shift_value(m_node.start(m_d, 0), m_shift);
        }
        else
        {
            m_current_value[m_d] = std::numeric_limits<coord_index_t>::max();
        }
        // spdlog::debug("RESET: dim = {}, level = {}, current_value = {}, start
        // = {}, end = {}", m_d, m_node.level(), m_current_value[m_d],
        // m_start[m_d], m_end[m_d]);
    }

    /**
     * Check if scan is in the current interval for the dimension d
     * @param scan the value to check
     */
    template <class T>
    inline bool subset_node<T>::eval(coord_index_t scan, std::size_t /*dim*/) const
    {
        // Recall that we check if scan is inside an interval defined as [start,
        // end[. The end of the interval is not included.
        //
        // if the current_value is the start of the interval which means ipos =
        // 0 then if scan is lower than current_value, scan is not in the
        // interval.
        //
        // if the current_value is the end of the interval which means ipos = 1
        // then if scan is lower than current_value, scan is in the interval.
        return !((scan < m_current_value[m_d]) ^ m_ipos[m_d]) && is_valid();
    }

    template <class T>
    inline void subset_node<T>::decrement_dim(coord_index_t i)
    {
        // The set is already invalided
        if (m_current_value[m_d] == std::numeric_limits<coord_index_t>::max())
        {
            // We don't find any interval -> invalidate the data structure
            m_start[m_d - 1] = 0;
            m_end[m_d - 1]   = 0;

            m_current_value[m_d - 1] = std::numeric_limits<coord_index_t>::max();
            m_index[m_d - 1]         = std::numeric_limits<std::size_t>::max();
        }
        else
        {
            auto shift_i = detail::shift_value(i, -m_shift);
            // Shift the index i for the dimension d to the level of the node.
            // spdlog::debug("DECREMENT_DIM: dim = {}, level = {}, i = {},
            // shift_i = {}", m_d, m_node.level(), i, shift_i);
            if (m_shift >= 0)
            {
                // The level of this node is lower or equal to the
                // min(ref_level, common_level)
                //
                // There can only be one correspondence between i and shift_i.
                //
                // For example, i = 16 at ref_level = 4 and the node is at level
                // 2 Thus the shift_i is equal to 16 >> (4 - 2 = 2) = 4 (only
                // one value is possible). In the same way, i = 18 at ref_level
                // = 4 and the node is at level 2 shift_i is again equal to 4.
                //
                //   16   17   18   19
                // |----|----|----|----| at level 4 (min(ref_level,
                // common_level))
                //
                //      8         9
                // |---------|---------| at level 3
                //
                //           4
                // |-------------------| at level 2 (node level)
                //

                // Check if we find this index in the list of intervals between
                // m_start_offset and m_end_offset spdlog::debug("DECREMENT_DIM:
                // start_offset = {}, end_offset = {}, i transformed = {}",
                // m_start_offset[m_d], m_end_offset[m_d], m_node.transform(m_d,
                // shift_i));
                std::size_t index = m_node.find(m_d, m_start_offset[m_d], m_end_offset[m_d], m_node.transform(m_d, shift_i));
                // spdlog::debug("DECREMENT_DIM: index found = {}", index);
                if (index != std::numeric_limits<std::size_t>::max())
                {
                    // We find an interval where shift_i is in.
                    auto interval = m_node.interval(m_d, index);

                    // Find the list of intervals for the dimension d - 1 for
                    // shift_i
                    auto off_ind     = static_cast<std::size_t>(interval.index + m_node.transform(m_d, shift_i));
                    m_start[m_d - 1] = m_node.offset(m_d, off_ind);
                    m_end[m_d - 1]   = m_node.offset(m_d, off_ind + 1);
                    // spdlog::debug("DECREMENT_DIM: start_offset = {}, off_ind
                    // = {} interval = {}", m_start_offset[m_d - 1], off_ind,
                    // interval);
                    m_start_offset[m_d - 1] = m_node.offset(m_d, off_ind);
                    m_end_offset[m_d - 1]   = m_node.offset(m_d, off_ind + 1);

                    // Initialize the current_value for dimension d - 1 with the
                    // start value of the first interval shifted to the
                    // min(ref_level, common_level)
                    m_current_value[m_d - 1] = detail::shift_value(m_node.start(m_d - 1, m_start[m_d - 1]), m_shift);
                    m_index[m_d - 1]         = m_start[m_d - 1];
                }
                else
                {
                    // We don't find any interval -> invalidate the data
                    // structure
                    m_start[m_d - 1] = 0;
                    m_end[m_d - 1]   = 0;

                    m_current_value[m_d - 1] = std::numeric_limits<coord_index_t>::max();
                    m_index[m_d - 1]         = std::numeric_limits<std::size_t>::max();
                }
            }
            else
            {
                // The level of this node is greater than the min(ref_level,
                // common_level)
                //
                // There can be multiple correspondences between i and shift_i.
                //
                // For example, i = 4 at ref_level = 2 and the node is at level
                // 4 Thus the shift_i can take multiple values x >> (4 - 2 = 2)
                // = 4. So, x can be equal to 16, 17, 18, 19.
                //
                //
                //           4
                // |-------------------| at level 2 (min(ref_level,
                // common_level))
                //
                //      8         9
                // |---------|---------| at level 3
                //
                //   16   17   18   19
                // |----|----|----|----| at level 4 (node level)
                //
                //
                // The idea is to build a new list of intervals for the d - 1
                // projected at min(ref_level, common_level) with all the
                // possible values for d
                //
                // Example:
                //
                // y: 16 -> x: [0, 6[, [8, 10[
                // y: 18 -> x: [10, 16[
                // y: 19 -> x: [-5, -3[, [26, 27[
                //
                // The new list of intervals is for y: 4 -> x: [-2, 5[, [6, 7[

                ListOfIntervals<coord_index_t, index_t> intervals;

                if (m_d == dim - 1)
                {
                    m_work_offsets[m_d - 1].clear();
                    for (int s = 0; s < 1 << (-m_shift); ++s) // NOLINT(hicpp-signed-bitwise)
                    {
                        std::size_t index = m_node.find(m_d, m_start_offset[m_d], m_end_offset[m_d], m_node.transform(m_d, shift_i + s));
                        if (index != std::numeric_limits<std::size_t>::max())
                        {
                            auto interval = m_node.interval(m_d, index);
                            auto off_ind  = static_cast<std::size_t>(interval.index + m_node.transform(m_d, shift_i + s));
                            m_work_offsets[m_d - 1].push_back(std::make_pair(m_node.offset(m_d, off_ind), m_node.offset(m_d, off_ind + 1)));

                            for (std::size_t o = m_node.offset(m_d, off_ind); o < m_node.offset(m_d, off_ind + 1); ++o)
                            {
                                auto start = m_node.start(m_d - 1, o) >> (-m_shift);
                                auto end   = ((m_node.end(m_d - 1, o) - 1) >> -m_shift) + 1;
                                if (start == end)
                                {
                                    end++;
                                }
                                intervals.add_interval({start, end});
                            }
                        }
                    }
                }
                else
                {
                    m_work_offsets[m_d - 1].clear();

                    for (auto& offset : m_work_offsets[m_d])
                    {
                        for (int s = 0; s < 1 << (-m_shift); ++s) // NOLINT(hicpp-signed-bitwise)
                        {
                            std::size_t index = m_node.find(m_d, offset.first, offset.second, m_node.transform(m_d, shift_i + s));

                            if (index != std::numeric_limits<std::size_t>::max())
                            {
                                auto interval = m_node.interval(m_d, index);
                                auto off_ind  = static_cast<std::size_t>(interval.index + m_node.transform(m_d, shift_i + s));
                                m_work_offsets[m_d - 1].push_back(
                                    std::make_pair(m_node.offset(m_d, off_ind), m_node.offset(m_d, off_ind + 1)));

                                for (std::size_t o = m_node.offset(m_d, off_ind); o < m_node.offset(m_d, off_ind + 1); ++o)
                                {
                                    auto start = m_node.start(m_d - 1, o) >> (-m_shift);
                                    auto end   = ((m_node.end(m_d - 1, o) - 1) >> -m_shift) + 1;
                                    if (start == end)
                                    {
                                        end++;
                                    }
                                    intervals.add_interval({start, end});
                                }
                            }
                        }
                    }
                }
                // spdlog::debug("intervals -> {}", intervals);
                m_work[m_d - 1].clear();

                if (intervals.size() != 0)
                {
                    std::copy(intervals.cbegin(), intervals.cend(), std::back_inserter(m_work[m_d - 1]));
                    m_start[m_d - 1]         = 0;
                    m_end[m_d - 1]           = m_work[m_d - 1].size();
                    m_current_value[m_d - 1] = m_work[m_d - 1][0].start;
                    m_index[m_d - 1]         = m_start[m_d - 1];
                }
                else
                {
                    m_start[m_d - 1] = 0;
                    m_end[m_d - 1]   = 0;

                    m_current_value[m_d - 1] = std::numeric_limits<coord_index_t>::max();
                    m_index[m_d - 1]         = std::numeric_limits<std::size_t>::max();
                }
            }
        }
        // spdlog::debug("For dimension {}, curent_value in decrement = {}", m_d
        // - 1, m_current_value[m_d - 1]);

        m_ipos[m_d - 1] = 0;
        m_d--;
    }

    template <class T>
    inline void subset_node<T>::increment_dim()
    {
        m_d++;
    }

    template <class T>
    inline void subset_node<T>::update(coord_index_t scan, coord_index_t sentinel)
    {
        // spdlog::debug("BEGIN UPDATE
        // ****************************************************************");
        if (scan == m_current_value[m_d])
        {
            // spdlog::debug("UPDATE: scan == current_value");
            if (m_ipos[m_d] == 1)
            {
                ++m_index[m_d];
                m_ipos[m_d] = 0;
                if (m_shift >= 0 || m_d == (dim - 1))
                {
                    m_current_value[m_d] = (m_index[m_d] >= m_end[m_d] ? sentinel
                                                                       : detail::shift_value(m_node.start(m_d, m_index[m_d]), m_shift));
                }
                else
                {
                    m_current_value[m_d] = (m_index[m_d] >= m_end[m_d] ? sentinel : m_work[m_d][m_index[m_d]].start);
                }
                // spdlog::debug("UPDATE: dim = {}, level = {}, start new
                // interval with current_value = {}", m_d, m_node.level(),
                // m_current_value[m_d]);
            }
            else
            {
                if (m_shift >= 0)
                {
                    m_current_value[m_d] = detail::shift_value(m_node.end(m_d, m_index[m_d]), m_shift);
                }
                else
                {
                    if (m_d == dim - 1)
                    {
                        coord_index_t value = detail::shift_value(m_node.end(m_d, m_index[m_d]) - 1, m_shift) + 1;
                        if (m_current_value[m_d] == value)
                        {
                            ++value;
                        }
                        // spdlog::debug("UPDATE: dim = {}, level = {}, value =
                        // {}, m_index = {}", m_d, m_node.level(), value,
                        // m_index[m_d]);
                        while (m_index[m_d] + 1 < m_end[m_d])
                        {
                            coord_index_t start_value = detail::shift_value(m_node.start(m_d, m_index[m_d] + 1), m_shift);
                            if (value >= start_value)
                            {
                                m_index[m_d]++;
                                value = detail::shift_value(m_node.end(m_d, m_index[m_d]) - 1, m_shift) + 1;
                                // spdlog::debug("UPDATE: dim = {}, level = {},
                                // value = {}, m_index = {}", m_d,
                                // m_node.level(), value, m_index[m_d]);
                            }
                            else
                            {
                                break;
                            }
                        }
                        m_current_value[m_d] = value;
                    }
                    else
                    {
                        m_current_value[m_d] = m_work[m_d][m_index[m_d]].end;
                    }
                }
                // spdlog::debug("UPDATE: dim = {}, level = {}, end interval
                // with current_value = {}", m_d, m_node.level(),
                // m_current_value[m_d]);
                m_ipos[m_d] = 1;
            }
        }
        // spdlog::debug("END UPDATE
        // ******************************************************************");
    }

    template <class T>
    inline auto subset_node<T>::min() const -> coord_index_t
    {
        return m_current_value[m_d];
    }

    template <class T>
    inline auto subset_node<T>::max() const -> coord_index_t
    {
        if (m_start[m_d] != m_end[m_d])
        {
            if (m_shift >= 0)
            {
                return detail::shift_value(m_node.end(m_d, m_end[m_d] - 1) + (m_node.end(m_d, m_end[m_d] - 1) & 1), m_shift);
            }

            if (m_d == (dim - 1))
            {
                return detail::shift_value(m_node.end(m_d, m_end[m_d] - 1) - 1, m_shift) + 1;
            }

            if (m_work[m_d].size() != 0)
            {
                return m_work[m_d].back().end;
            }
        }
        return std::numeric_limits<coord_index_t>::min();
    }

    template <class T>
    inline std::size_t subset_node<T>::common_level() const
    {
        return m_node.level();
    }

    template <class T>
    inline void subset_node<T>::set_shift(std::size_t ref_level, std::size_t common_level)
    {
        m_ref_level    = ref_level;
        m_common_level = common_level;
        m_shift_ref    = safe_subs<int>(ref_level, m_node.level());
        m_shift_common = safe_subs<int>(common_level, m_node.level());
        m_shift        = std::min(m_shift_ref, m_shift_common);
        // spdlog::debug("SET SHIFT: level = {}, ref_level = {}, common_level =
        // {}, shift_ref = {}, shift_common = {}, shift = {}", m_node.level(),
        // m_ref_level, m_common_level, m_shift_ref, m_shift_common, m_shift);
    }

    template <class T>
    inline auto subset_node<T>::get_node() const -> const node_type&
    {
        return m_node;
    }

    template <class T>
    inline void subset_node<T>::get_interval_index(std::vector<std::size_t>& index) const
    {
        index.push_back(m_index[m_d] + m_ipos[m_d] - 1);
    }
} // namespace samurai
