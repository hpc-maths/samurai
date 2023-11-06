#pragma once

#include <xtensor/xarray.hpp>

#include <samurai/interval.hpp>
#include <samurai/list_of_intervals.hpp>

namespace samurai
{
    template <typename coord_t, typename index_t>
    bool operator==(const ListOfIntervals<coord_t, index_t>& li, const xt::xarray<Interval<coord_t, index_t>>& array)
    {
        auto ix = li.cbegin();
        auto iy = array.cbegin();
        while (ix != li.cend() && iy != array.cend())
        {
            if (*ix != *iy)
            {
                return false;
            }
            ++ix;
            ++iy;
        }
        if (ix == li.cend() && iy == array.cend())
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}
