#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <list>
#include <type_traits>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include "interval.hpp"

namespace mure
{
    template<typename MRConfig>
    class LevelCellList
    {
    public:

        static constexpr auto dim = MRConfig::dim;
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        using list_interval_t = ListOfIntervals<coord_index_t, index_t>;

        void extend(xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> start,
                    xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> end)
        {
            if (xt::all(start < end))
            {
                auto size = end - start;
                // we have data
                if (dim != 1 && m_box_yz.isvalid())
                {
                    xt::xtensor<list_interval_t, dim-1> new_grid;
                    new_grid.resize(xt::eval(size));
                    xt::xstrided_slice_vector sv;
                    for(std::size_t i=0; i<dim-1; ++i)
                        sv.push_back(xt::range(static_cast<std::size_t>(start[i]-m_box_yz.min_corner()[i]),
                                               static_cast<std::size_t>(end[i]-m_box_yz.max_corner()[i])));
                    auto view = xt::strided_view(new_grid, sv);
                    xt::noalias(view) = m_grid_yz;
                    std::swap(m_grid_yz, new_grid);
                }
                else{
                    m_grid_yz.resize(xt::eval(size));
                }
                m_box_yz = {start, end};
            }
        }

        void fill(interval_t interval)
        {
            std::fill(m_grid_yz.begin(), m_grid_yz.end(), list_interval_t {interval});
        }

        inline typename Box<coord_index_t, dim-1>::point_t const& min_corner_yz() const
        {
            return m_box_yz.min_corner();
        }

        inline typename Box<coord_index_t, dim-1>::point_t const& max_corner_yz() const
        {
            return m_box_yz.max_corner();
        }

        list_interval_t const& operator[](xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index) const
        {
            return m_grid_yz[index - m_box_yz.min_corner()];
        }

        list_interval_t& operator[](xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index)
        {
            return m_grid_yz[index - m_box_yz.min_corner()];
        }

        void to_stream(std::ostream &os) const
        {
            os << "LevelCellList\n";
            os << "=============\n";
            os << m_box_yz << "\n";
            os << m_grid_yz << "\n";
        }
    private:
        xt::xtensor<list_interval_t, dim-1> m_grid_yz;
        Box<coord_index_t, dim-1> m_box_yz;
    };

    template<class MRConfig>
    std::ostream& operator<<(std::ostream& out, const LevelCellList<MRConfig>& level_cell_list)
    {
        level_cell_list.to_stream(out);
        return out;
    }
}