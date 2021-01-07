#pragma once

#include <xtensor/xarray.hpp>

#include <rapidcheck.h>

#include <samurai/interval.hpp>
#include <samurai/list_of_intervals.hpp>

namespace samurai
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

// NOTE: Must be in rc namespace!
namespace rc
{

    template<class TValue>
    struct Arbitrary<samurai::Interval<TValue>>
    {
        static Gen<samurai::Interval<TValue>> arbitrary()
        {
            auto start = gen::inRange(-100, 100);
            auto end = gen::inRange(-100, 100);
            return gen::build<samurai::Interval<TValue>>(
                gen::set(&samurai::Interval<TValue>::start, start),
                gen::set(&samurai::Interval<TValue>::end, end));
        }
    };

} // namespace rc