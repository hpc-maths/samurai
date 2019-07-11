#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>

#include <mure/cell.hpp>

namespace mure
{
    TEST(cell, length)
    {
        Cell<int, 2> c{1, {1, 1}, 0};
        EXPECT_EQ(c.length(), 0.5);
    }

    TEST(cell, center)
    {
        Cell<int, 2> c{1, {1, 1}, 0};
        xt::xarray<double> expected{.75, .75};
        EXPECT_EQ(c.center(), expected);
    }

    TEST(cell, first_corner)
    {
        Cell<int, 2> c{1, {1, 1}, 0};
        xt::xarray<double> expected{.5, .5};
        EXPECT_EQ(c.first_corner(), expected);
    }
}