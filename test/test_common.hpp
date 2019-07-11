#pragma once

#include <xtensor/xarray.hpp>

#include <mure/interval.hpp>
#include <mure/list_of_intervals.hpp>

namespace mure
{
    template<typename coord_t, typename index_t>
    bool operator==(ListOfIntervals<coord_t, index_t> const &li,
                    xt::xarray<Interval<coord_t, index_t>> const &array)
    {
        auto ix = li.cbegin();
        auto iy = array.cbegin();
        while (ix != li.cend() && iy != array.cend())
        {
            if (*ix != *iy)
                return false;
            ++ix;
            ++iy;
        }
        if (ix == li.cend() && iy == array.cend())
            return true;
        else
            return false;
    }
}
