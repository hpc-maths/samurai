#pragma once

#include <algorithm>
#include <array>
#include <iterator>

// #include <xtensor/xadapt.hpp>
// #include <xtensor/xio.hpp>
// #include <spdlog/spdlog.h>
// #include <spdlog/fwd.h>

#include "node_op.hpp"

namespace mure
{
    namespace detail
    {
        inline int shift_value(int value, int shift)
        {
            return (shift >= 0)? (value << shift): (value >> (-shift));
        }
    }
    /**************************
     * subset_node definition *
     **************************/

    template<class T>
    class subset_node {
      public:
        using node_type = T;
        using mesh_type = typename node_type::mesh_type;
        static constexpr std::size_t dim = mesh_type::dim;
        using interval_t = typename mesh_type::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using index_t = typename interval_t::index_t;

        subset_node(T &&node);

        bool is_valid() const;
        bool is_empty() const;
        void reset();
        auto eval(coord_index_t scan, std::size_t dim) const;
        bool is_in(coord_index_t scan) const;
        void decrement_dim(coord_index_t i);
        void increment_dim();
        void update(coord_index_t scan, coord_index_t sentinel);
        coord_index_t min() const;
        coord_index_t max() const;
        std::size_t common_level() const;
        void set_shift(std::size_t ref_level, std::size_t common_level);
        const node_type &get_node() const;

        void get_interval_index(std::vector<std::size_t> &index)
        {
            index.push_back(m_index[m_d] + m_ipos[m_d] - 1);
        }

      private:
        int m_shift;
        int m_shift_ref;
        int m_shift_common;
        std::size_t m_ref_level;
        std::size_t m_common_level;
        std::size_t m_d;
        std::array<std::size_t, dim> m_index;
        std::array<std::size_t, dim> m_ipos;
        std::array<std::size_t, dim> m_start;
        std::array<std::size_t, dim> m_end;
        std::array<std::size_t, dim> m_start_offset;
        std::array<std::size_t, dim> m_end_offset;
        std::array<coord_index_t, dim> m_current_value;
        std::array<std::vector<interval_t>, dim> m_work;
        node_type m_node;
    };

    /******************************
     * subset_node implementation *
     ******************************/

    template<class T>
    inline subset_node<T>::subset_node(T &&node) : m_node(std::forward<T>(node))
    {}

    template<class T>
    inline bool subset_node<T>::is_valid() const
    {
        return !(m_end == m_start);
    }

    template<class T>
    inline bool subset_node<T>::is_empty() const
    {
        return m_node.is_empty();
    }

    template<class T>
    inline void subset_node<T>::reset()
    {
        m_d = dim - 1;
        m_start[m_d] = 0;
        m_end[m_d] = m_node.size(m_d);
        m_start_offset[m_d] = 0;
        m_end_offset[m_d] = m_node.size(m_d);
        m_index[m_d] = m_start[m_d];
        m_ipos[m_d] = 0;
        if (m_start[m_d] != m_end[m_d])
        {
            m_current_value[m_d] = detail::shift_value(m_node.start(m_d, 0), m_shift);
        }
        else
        {
            m_current_value[m_d] = std::numeric_limits<coord_index_t>::max();
        }
        //spdlog::debug("RESET: dim = {}, level = {}, current_value = {}, start = {}, end = {}", m_d, m_node.level(), m_current_value[m_d], m_start[m_d], m_end[m_d]);
    }

    template<class T>
    inline auto subset_node<T>::eval(coord_index_t scan, std::size_t /*dim*/) const
    {
        return is_in(scan);
    }

    template<class T>
    inline bool subset_node<T>::is_in(coord_index_t scan) const
    {
        return !((scan < m_current_value[m_d]) ^ m_ipos[m_d]) & is_valid();
    }

    template<class T>
    inline void subset_node<T>::decrement_dim(coord_index_t i)
    {
        std::size_t index;
        auto shift_i = detail::shift_value(i, -m_shift);
        //spdlog::debug("DECREMENT_DIM: level = {}, i = {}, shift_i = {}", m_node.level(), i, shift_i);
        if (m_shift >= 0)
        {
            index = m_node.find(m_d, m_start_offset[m_d], m_end_offset[m_d], m_node.transform(m_d, shift_i));
            if (index != std::numeric_limits<std::size_t>::max())
            {
                auto interval = m_node.interval(m_d, index);
                std::size_t off_ind = interval.index + m_node.transform(m_d, shift_i);
                m_start[m_d - 1] = m_node.offset(m_d, off_ind);
                m_end[m_d - 1] = m_node.offset(m_d, off_ind + 1);
                m_start_offset[m_d - 1] = m_node.offset(m_d, off_ind);
                m_end_offset[m_d - 1] = m_node.offset(m_d, off_ind + 1);
                m_current_value[m_d - 1] = detail::shift_value(m_node.start(m_d - 1, m_start[m_d - 1]), m_shift);
                m_index[m_d - 1] = m_start[m_d - 1];
            }
            else
            {
                m_start[m_d - 1] = 0;
                m_end[m_d - 1] = 0;

                m_current_value[m_d - 1] = std::numeric_limits<coord_index_t>::max();
                m_index[m_d - 1] = std::numeric_limits<std::size_t>::max();
            }
        }
        else
        {
            ListOfIntervals<coord_index_t, index_t> intervals;
            m_work[m_d - 1].clear();
            for(std::size_t s=0; s< (1<<(-m_shift)); ++s)
            {
                index = m_node.find(m_d, m_start_offset[m_d], m_end_offset[m_d], m_node.transform(m_d, shift_i + s));
                bool first_found = true;
                if (index != std::numeric_limits<std::size_t>::max())
                {
                    auto interval = m_node.interval(m_d, index);
                    std::size_t off_ind = interval.index + m_node.transform(m_d, shift_i + s);
                    if (first_found)
                    {
                        m_start_offset[m_d - 1] = m_node.offset(m_d, off_ind);
                        first_found = false;
                    }

                    for(coord_index_t o=m_node.offset(m_d, off_ind); o<m_node.offset(m_d, off_ind+1); ++o)
                    {
                        auto start = m_node.start(m_d - 1, o)>>(-m_shift);
                        auto end = (m_node.end(m_d - 1, o) + (m_node.end(m_d - 1, o) & 1))>>(-m_shift);
                        if (start == end)
                        {
                            end++;
                        }
                        intervals.add_interval({start, end});
                    }
                    m_end_offset[m_d - 1] = m_node.offset(m_d, off_ind + 1);
                }
            }
            //spdlog::debug("intervals -> {}", intervals);
            if (intervals.size() != 0)
            {
                std::copy(intervals.cbegin(), intervals.cend(), std::back_inserter(m_work[m_d - 1]));
                m_start[m_d - 1] = 0;
                m_end[m_d - 1] = m_work[m_d - 1].size();
                m_current_value[m_d - 1] = m_work[m_d - 1][0].start;
                m_index[m_d - 1] = m_start[m_d - 1];
            }
            else
            {
                m_start[m_d - 1] = 0;
                m_end[m_d - 1] = 0;

                m_current_value[m_d - 1] = std::numeric_limits<coord_index_t>::max();
                m_index[m_d - 1] = std::numeric_limits<std::size_t>::max();
            }
        }
        //spdlog::debug("For dimension {}, curent_value in decrement = {}", m_d - 1, m_current_value[m_d - 1]);

        m_ipos[m_d - 1] = 0;
        m_d--;
    }

    template<class T>
    inline void subset_node<T>::increment_dim()
    {
        m_d++;
    }

    template<class T>
    inline void subset_node<T>::update(coord_index_t scan, coord_index_t sentinel)
    {
        //spdlog::debug("BEGIN UPDATE ****************************************************************");
        if (scan == m_current_value[m_d])
        {
            //spdlog::debug("UPDATE: scan == current_value");
            if (m_ipos[m_d] == 1)
            {
                m_index[m_d] += 1;
                m_ipos[m_d] = 0;
                if (m_shift >= 0 or m_d == (dim - 1))
                {
                    m_current_value[m_d] = (m_index[m_d] >= m_end[m_d]
                                                ? sentinel
                                                : detail::shift_value(m_node.start(m_d, m_index[m_d]), m_shift));
                }
                else
                {
                    m_current_value[m_d] = (m_index[m_d] >= m_end[m_d]
                                                ? sentinel
                                                : m_work[m_d][m_index[m_d]].start);
                }
                //spdlog::debug("UPDATE: dim = {}, level = {}, start new interval with current_value = {}", m_d, m_node.level(), m_current_value[m_d]);
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
                        coord_index_t value = detail::shift_value(m_node.end(m_d, m_index[m_d]) + (m_node.end(m_d, m_index[m_d]) & 1), m_shift);
                        if (m_current_value[m_d] == value)
                        {
                            value++;
                        }
                        while (m_index[m_d] + 1 < m_end[m_d])
                        {
                            coord_index_t start_value = detail::shift_value(m_node.start(m_d, m_index[m_d] + 1), m_shift);
                            if (value == start_value)
                            {
                                m_index[m_d]++;
                                value = detail::shift_value(m_node.end(m_d, m_index[m_d]) + (m_node.end(m_d, m_index[m_d]) & 1), m_shift);
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
                //spdlog::debug("UPDATE: dim = {}, level = {}, end interval with current_value = {}", m_d, m_node.level(), m_current_value[m_d]);
                m_ipos[m_d] = 1;
            }
        }
        //spdlog::debug("END UPDATE ******************************************************************");
    }

    template<class T>
    inline auto subset_node<T>::min() const -> coord_index_t
    {
        return m_current_value[m_d];
    }

    template<class T>
    inline auto subset_node<T>::max() const -> coord_index_t
    {
        if (m_start[m_d] != m_end[m_d])
        {
            if (m_shift >= 0 or m_d == (dim - 1))
            {
                return detail::shift_value(m_node.end(m_d, m_end[m_d] - 1), m_shift);
            }
            else
            {
                if (m_work[m_d].size() != 0)
                {
                    return m_work[m_d].back().end;
                }
                else
                {
                    return std::numeric_limits<coord_index_t>::min();
                }
            }
        }
        else
        {
            return std::numeric_limits<coord_index_t>::min();
        }
    }

    template<class T>
    inline std::size_t subset_node<T>::common_level() const
    {
        return m_node.level();
    }

    template<class T>
    inline void subset_node<T>::set_shift(std::size_t ref_level, std::size_t common_level)
    {
        m_ref_level = ref_level;
        m_common_level = common_level;
        m_shift_ref = ref_level - m_node.level();
        m_shift_common = common_level - m_node.level();
        m_shift = std::min(m_shift_ref, m_shift_common);
        //spdlog::debug("SET SHIFT: level = {}, ref_level = {}, common_level = {}, shift_ref = {}, shift_common = {}, shift = {}", m_node.level(), m_ref_level, m_common_level, m_shift_ref, m_shift_common, m_shift);
    }

    template<class T>
    inline auto subset_node<T>::get_node() const -> const node_type &
    {
        return m_node;
    }
}