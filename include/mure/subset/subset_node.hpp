#pragma once

#include <array>

#include "node_op.hpp"

namespace mure
{
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
        using coord_index_t = typename interval_t::value_t;

        subset_node(T &&node);

        bool is_valid() const;
        void reset();
        auto eval(int scan, std::size_t dim) const;
        bool is_in(int scan) const;
        void decrement_dim(int i);
        void increment_dim();
        void update(coord_index_t scan, coord_index_t sentinel);
        int min() const;
        int max() const;
        const node_type &get_node() const;

      private:
        std::size_t m_d;
        std::array<std::size_t, dim> m_index;
        std::array<std::size_t, dim> m_ipos;
        std::array<std::size_t, dim> m_start;
        std::array<std::size_t, dim> m_end;
        std::array<coord_index_t, dim> m_current_value;
        node_type m_node;
    };

    /******************************
     * subset_node implementation *
     ******************************/

    template<class T>
    inline subset_node<T>::subset_node(T &&node) : m_node(std::forward<T>(node))
    {
        reset();
    }

    template<class T>
    inline bool subset_node<T>::is_valid() const
    {
        return !(m_end == m_start);
    }

    template<class T>
    inline void subset_node<T>::reset()
    {
        m_d = dim - 1;
        m_start[m_d] = 0;
        m_end[m_d] = m_node.size(m_d);
        m_index[m_d] = m_start[m_d];
        m_ipos[m_d] = 0;
        m_current_value[m_d] = m_node.start(m_d, 0);
    }

    template<class T>
    inline auto subset_node<T>::eval(int scan, std::size_t /*dim*/) const
    {
        return is_in(scan);
    }

    template<class T>
    inline bool subset_node<T>::is_in(int scan) const
    {
        return !((scan < m_current_value[m_d]) ^ m_ipos[m_d]) & is_valid();
    }

    template<class T>
    inline void subset_node<T>::decrement_dim(int i)
    {
        int index = m_index[m_d] + m_ipos[m_d] - 1;
        auto interval = m_node.interval(m_d, index);
        int off_ind = (index != -1) ? interval.index + m_node.index(i)
                                    : std::numeric_limits<int>::max();

        if (off_ind < interval.index + interval.start)
            off_ind = interval.index + interval.start;

        m_start[m_d - 1] =
            (index != -1 and
             off_ind < static_cast<int>(m_node.offsets_size(m_d)))
                ? m_node.offset(m_d, off_ind)
                : 0;

        m_end[m_d - 1] =
            (index != -1 and
             (off_ind + 1) < static_cast<int>(m_node.offsets_size(m_d)))
                ? m_node.offset(m_d, off_ind + 1)
                : m_start[m_d - 1];

        m_index[m_d - 1] = m_start[m_d - 1];
        m_current_value[m_d - 1] = m_node.start(m_d - 1, m_start[m_d - 1]);
        m_ipos[m_d - 1] = 0;
        m_d--;
    }

    template<class T>
    inline void subset_node<T>::increment_dim()
    {
        m_d++;
    }

    template<class T>
    inline void subset_node<T>::update(coord_index_t scan,
                                       coord_index_t sentinel)
    {
        if (scan == m_current_value[m_d])
        {
            if (m_ipos[m_d] == 1)
            {
                m_index[m_d] += 1;
                m_ipos[m_d] = 0;
                m_current_value[m_d] = (m_index[m_d] == m_end[m_d]
                                            ? sentinel
                                            : m_node.start(m_d, m_index[m_d]));
            }
            else
            {
                m_current_value[m_d] = m_node.end(m_d, m_index[m_d]);
                m_ipos[m_d] = 1;
            }
        }
    }

    template<class T>
    inline int subset_node<T>::min() const
    {
        return m_current_value[m_d];
    }

    template<class T>
    inline int subset_node<T>::max() const
    {
        if (m_start[m_d] != m_end[m_d])
        {
            return m_node.end(m_d, m_end[m_d] - 1);
        }
        else
        {
            return std::numeric_limits<coord_index_t>::min();
        }
    }

    template<class T>
    inline auto subset_node<T>::get_node() const -> const node_type &
    {
        return m_node;
    }
}