#pragma once

#include <array>
#include <deque>
#include <vector>
#include <ostream>

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include "box.hpp"
#include "interval.hpp"
#include "level_cell_list.hpp"

namespace mure
{
    template<class MRConfig>
    class LevelCellArray
    {
    public:

        static constexpr auto dim = MRConfig::dim;
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;

        LevelCellArray(LevelCellList<MRConfig> const& lcl);

        /// Display to the given stream
        void to_stream(std::ostream& out) const;

        template<class function_t>
        void for_each_interval_in_x(function_t&& f) const;

        bool empty() const;

        inline typename Box<coord_index_t, dim-1>::point_t const& min_corner_yz() const
        {
            return m_box_yz.min_corner();
        }

        inline typename Box<coord_index_t, dim-1>::point_t const& max_corner_yz() const
        {
            return m_box_yz.max_corner();
        }

        auto const& operator[](index_t index) const
        {
            return m_cells[index];
        }

        auto& operator[](index_t index)
        {
            return m_cells[index];
        }

        auto const& offset(index_t index) const
        {
            return m_offsets[index];
        }

        auto& offset(index_t index)
        {
            return m_offsets[index];
        }

        auto beg_ind_last_dim() const
        {
            return _beg_ind_last_dim;
        }

        auto size() const
        {
            return m_cells.size();
        }

    private:

        template<class function_t, index_t d>
        void for_each_interval_in_x_impl(function_t&& f,
                                         xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index_yz,
                                         index_t beg_index,
                                         index_t end_index,
                                         std::integral_constant<index_t, d>) const;

        template<class function_t>
        void for_each_interval_in_x_impl(function_t&& f,
                                         xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index_yz,
                                         index_t beg_index,
                                         index_t end_index,
                                         std::integral_constant<index_t, 0>) const;

        void set_box_yz();

        template<index_t d>
        void set_box_yz_impl(index_t start_index, index_t end_index,
                             std::integral_constant<index_t, d>);

        void set_box_yz_impl(index_t start_index, index_t end_index,
                             std::integral_constant<index_t, 1>);

        template<int d>
        void _get_inter_ranges(std::array<std::deque<index_t>, dim-1>& inter_sizes,
                               std::array<std::deque<interval_t>, dim>& inter_ranges,
                               LevelCellList<MRConfig> const& lcl,
                               std::integral_constant<int, d>,
                               xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index ) const;

        void _get_inter_ranges(std::array<std::deque<index_t>, dim-1>& inter_sizes,
                               std::array<std::deque<interval_t>, dim>& inter_ranges,
                               LevelCellList<MRConfig> const& lcl,
                               std::integral_constant<int, 1>,
                               xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index ) const;

        std::vector<Interval<coord_index_t, index_t>> m_cells;
        std::vector<index_t> m_offsets;
        Box<coord_index_t, dim-1> m_box_yz;
        index_t _beg_ind_last_dim;
        index_t _end_ind_x_ranges;
    };

    template<typename MRConfig>
    LevelCellArray<MRConfig>::LevelCellArray(LevelCellList<MRConfig> const& lcl)
    {
        std::array<std::deque<index_t>, dim-1> inter_sizes;
        std::array<std::deque<interval_t>, dim> inter_ranges;

        _get_inter_ranges(inter_sizes, inter_ranges, lcl,
                            std::integral_constant<int, dim>{},
                            {});

        // reservations
        index_t nb_ranges = 0;
        for(index_t d = 0; d < dim; ++d)
            nb_ranges += inter_ranges[d].size();
        m_cells.reserve(nb_ranges);

        index_t nb_offsets = 1;
        for(index_t d = 0; d < dim - 1; ++d)
            nb_offsets += inter_sizes[d].size();
        m_offsets.reserve(nb_offsets);

        // ranges for x
        for(auto const& r : inter_ranges[0])
            m_cells.emplace_back(r.start, r.end);

        // ranges and offsets for y, z, ...
        index_t acc_range = 0;
        index_t acc_offset = 0;
        m_offsets.emplace_back(0);
        for(index_t d = 1; d < dim; ++d)
        {
            for(auto const& r: inter_ranges[d] )
            {
                m_cells.emplace_back(r.start, r.end, acc_range - r.start);
                acc_range += r.size();
            }
            for(index_t size : inter_sizes[d - 1])
                m_offsets.emplace_back(acc_offset += size);
        }

        // needed offsets
        _beg_ind_last_dim = m_cells.size() - inter_ranges[dim - 1].size();
        _end_ind_x_ranges = inter_ranges[0].size();

        set_box_yz();
    }

    template<class MRConfig>
    template<class function_t>
    void LevelCellArray<MRConfig>::for_each_interval_in_x(function_t&& f) const
    {
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index_yz;
        for_each_interval_in_x_impl(std::forward<function_t>(f),
                                    index_yz,
                                    _beg_ind_last_dim,
                                    m_cells.size(),
                                    std::integral_constant<index_t, dim-1>{});
    }

    template<class MRConfig>
    template<class function_t>
    void LevelCellArray<MRConfig>::for_each_interval_in_x_impl(function_t&& f,
                                                          xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index_yz,
                                                          index_t beg_index,
                                                          index_t end_index,
                                                          std::integral_constant<index_t, 0>) const
    {
        for(index_t i = beg_index; i < end_index; ++i)
        {
            std::forward<function_t>(f)(index_yz, m_cells[i]);
        }
    }

    template<class MRConfig>
    template<class function_t, typename MRConfig::index_t d>
    void LevelCellArray<MRConfig>::for_each_interval_in_x_impl(function_t&& f,
                                                          xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index_yz,
                                                          index_t beg_index,
                                                          index_t end_index,
                                                          std::integral_constant<index_t, d>) const
    {
        for(index_t i = beg_index; i < end_index; ++i)
        {
            for(coord_index_t c = m_cells[i].start; c < m_cells[i].end; ++c )
            {
                index_yz[d] = c;
                for_each_interval_in_x_impl(std::forward<function_t>(f),
                                            index_yz,
                                            m_offsets[m_cells[i].index + c],
                                            m_offsets[m_cells[i].index + c + 1],
                                            std::integral_constant<index_t, d-1>{});
            }
        }
    }

    template<class MRConfig>
    bool LevelCellArray<MRConfig>::empty() const
    {
        return m_cells.empty();
    }

    template<class MRConfig>
    void LevelCellArray<MRConfig>::set_box_yz()
    {
        if (dim > 1)
        {
            m_box_yz.min_corner().fill(std::numeric_limits<coord_index_t>::max());
            m_box_yz.max_corner().fill(std::numeric_limits<coord_index_t>::min());
            set_box_yz_impl(_beg_ind_last_dim, m_cells.size(),
                            std::integral_constant<index_t, dim>{});
        }
    }

    template<class MRConfig>
    template<typename MRConfig::index_t d>
    void LevelCellArray<MRConfig>::set_box_yz_impl(index_t start_index, index_t end_index,
                                                   std::integral_constant<index_t, d>)
    {
        if (start_index == end_index)
        {
            return;
        }

        m_box_yz.min_corner()[d-2] = std::min(m_box_yz.min_corner()[d-2], m_cells[start_index].start);
        m_box_yz.max_corner()[d-2] = std::max(m_box_yz.max_corner()[d-2], m_cells[end_index-1].end );

        if (d >= 3)
        {
            for(index_t i = start_index; i < end_index; ++i)
            {
                for(coord_index_t c = m_cells[i].start; c < m_cells[i].end; ++c)
                {
                    set_box_yz_impl(m_offsets[m_cells[i].index + c], m_offsets[m_cells[i].index + c + 1],
                                    std::integral_constant<index_t, d-1>{});
                }
            }
        }
    }

    template<class MRConfig>
    void LevelCellArray<MRConfig>::set_box_yz_impl(index_t beg_index, index_t end_index,
                                            std::integral_constant<index_t, 1>)
    {}

    template<typename MRConfig>
    template<int d>
    void LevelCellArray<MRConfig>::_get_inter_ranges(std::array<std::deque<index_t>, dim-1>& inter_sizes,
                                                     std::array<std::deque<interval_t>, dim>& inter_ranges,
                                                     LevelCellList<MRConfig> const& lcl,
                                                     std::integral_constant<int, d>,
                                                     xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index) const
    {
        interval_t new_interval = { 1, 0 };
        for(coord_index_t c = lcl.min_corner_yz()[d-2]; c < lcl.max_corner_yz()[d-2]; ++c)
        {
            index_t old_nb_ranges = inter_ranges[d-2].size();
            index[d-2] = c;
            _get_inter_ranges(inter_sizes, inter_ranges, lcl,
                              std::integral_constant<int, d-1>{}, index);

            if (index_t size = inter_ranges[d-2].size() - old_nb_ranges)
            {
                if ( new_interval.is_valid() )
                {
                    if ( new_interval.end == c )
                    {
                        new_interval.end = c + 1;
                    }
                    else
                    {
                        inter_ranges[d-1].emplace_back(new_interval);
                        new_interval = {1, 0};
                    }
                }
                else
                {
                    new_interval = {c, c + 1};
                }

                inter_sizes[d-2].emplace_back(size);
            }
            else if (new_interval.is_valid())
            {
                inter_ranges[d-1].emplace_back(new_interval);
                new_interval = {1, 0};
            }
        }

        if (new_interval.is_valid())
        {
            inter_ranges[d-1].emplace_back(new_interval);
        }
    }

    template<class MRConfig>
    void LevelCellArray<MRConfig>::_get_inter_ranges(std::array<std::deque<index_t>, dim-1>& inter_sizes,
                                                     std::array<std::deque<interval_t>, dim>& inter_ranges,
                                                     LevelCellList<MRConfig> const& lcl,
                                                     std::integral_constant<int, 1>,
                                                     xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index) const
    {
        auto const& interval_list = lcl[index];
        for(auto const& interval: interval_list)
        {
            inter_ranges[0].emplace_back(interval.start, interval.end);
        }
    }

    template <class MRConfig>
    void
    LevelCellArray<MRConfig>::
    to_stream(std::ostream& out) const
    {
        out << "cells = ";
        for (auto const& interval : m_cells)
            std::cout << interval << " ";
        out << std::endl;

        out << "offsets = ";
        for (auto const& v : m_offsets)
            std::cout << v << " ";
        out << std::endl;

        out << "beg_ind_last_dim = " << _beg_ind_last_dim << std::endl;
        out << "end_ind_x_ranges = " << _end_ind_x_ranges << std::endl;
        out << "Box\t" << m_box_yz.min_corner() << " " << m_box_yz.max_corner() << std::endl;
    }

    template <class MRConfig>
    std::ostream& operator<< (std::ostream& out, LevelCellArray<MRConfig> const& level_cell_array)
    {
        level_cell_array.to_stream(out);
        return out;
    }


}

