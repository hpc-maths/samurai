#pragma once

namespace samurai
{
    namespace experimental
    {

        template <typename TInterval, typename TCoord>
        class ArrayOfIntervalAndPoint
        {
          public:

            using interval_t = TInterval;
            using coord_type = TCoord;

            using reference       = std::pair<interval_t&, coord_type&>;
            using const_reference = std::pair<const interval_t&, const coord_type&>;

            ArrayOfIntervalAndPoint()
            {
            }

            reference operator[](const size_t i)
            {
                return reference(m_intervals[m_idx[i]], m_coords[m_idx[i]]);
            }

            const_reference operator[](const size_t i) const
            {
                return const_reference(m_intervals[m_idx[i]], m_coords[m_idx[i]]);
            }

            interval_t& get_interval(const size_t i)
            {
                return m_intervals[m_idx[i]];
            }

            const interval_t& get_interval(const size_t i) const
            {
                return m_intervals[m_idx[i]];
            }

            coord_type& get_coord(const size_t i)
            {
                return m_coords[m_idx[i]];
            }

            const coord_type& get_coord(const size_t i) const
            {
                return m_coords[m_idx[i]];
            }

            size_t size() const
            {
                return m_idx.size();
            }

            void push_back(const interval_t& interval, const coord_type& coord)
            {
                m_intervals.push_back(interval);
                m_coords.push_back(coord);
                m_idx.push_back(m_coords.size() - 1);
            }

            void clear()
            {
                m_intervals.clear();
                m_coords.clear();
                m_idx.clear();
            }

            void sort_intervals();
            void remove_overlapping_intervals();

          private:

            void init_indices()
            {
                m_idx.resize(m_coords.size());
                std::iota(m_idx.begin(), m_idx.end(), 0);
            }

            std::vector<interval_t> m_intervals;
            std::vector<coord_type> m_coords;
            std::vector<size_t> m_idx;
        };

        template <typename TInterval, typename TCoord>
        void ArrayOfIntervalAndPoint<TInterval, TCoord>::sort_intervals()
        {
            if (m_idx.size() != m_coords.size())
            {
                init_indices();
            }
            // sort the indices
            std::stable_sort(m_idx.begin(),
                             m_idx.end(),
                             [this](const size_t i1, const size_t i2) -> bool
                             {
                                 const auto& yz_lhs = m_coords[i1];
                                 const auto& yz_rhs = m_coords[i2];
                                 for (size_t i = yz_lhs.size() - 1; i != size_t(-1); --i)
                                 {
                                     if (yz_lhs[i] < yz_rhs[i])
                                     {
                                         return true;
                                     }
                                     else if (yz_lhs[i] > yz_rhs[i])
                                     {
                                         return false;
                                     }
                                 }
                                 return m_intervals[i1].start < m_intervals[i2].start;
                             });
        }

        // not needed now but it might be needed in the future so I rather keep it
        template <typename TInterval, typename TCoord>
        void ArrayOfIntervalAndPoint<TInterval, TCoord>::remove_overlapping_intervals()
        {
            sort_intervals();
            // remove idndices corresponding to duplicate entries
            const auto it = std::unique(m_idx.begin(),
                                        m_idx.end(),
                                        [this](const size_t i1, const size_t i2) -> bool
                                        {
                                            if (m_coords[i1] == m_coords[i2])
                                            { // we check for overlap, we know that intervals[i1].start <= intervals[i2].start
                                                if (m_intervals[i2].start <= m_intervals[i1].end) // there is an overlap
                                                {
                                                    m_intervals[i1].end = m_intervals[i2].end;
                                                    return true;
                                                } // there is no overlap just test if the intervals are the same
                                                return m_intervals[i1].start == m_intervals[i2].start; // don't need to test for equality of
                                                                                                       // end since it was tested above
                                            }
                                            return false;
                                        });
            m_idx.erase(it, m_idx.end());
        }

    }
}
