#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>

#include <samurai/cell.hpp>
#include <samurai/interval.hpp>

namespace samurai
{
    TEST(cell, length)
    {
        auto indices = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        Cell<2, Interval<int>> c{1, indices, 0};
        EXPECT_EQ(c.length, 0.5);
    }

    TEST(cell, center)
    {
        auto indices = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        Cell<2, Interval<int>> c{1, indices, 0};
        xt::xarray<double> expected{.75, .75};
        EXPECT_EQ(c.center(), expected);
    }

    TEST(cell, first_corner)
    {
        auto indices = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        Cell<2, Interval<int>> c{1, indices, 0};
        xt::xarray<double> expected{.5, .5};
        EXPECT_EQ(c.corner(), expected);
    }
}
