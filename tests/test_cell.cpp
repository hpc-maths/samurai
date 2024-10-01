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

    /**
     * Test of samurai::cell_length function (cell.hpp)
    */
    TEST(cell, cell_length){
        EXPECT_DOUBLE_EQ(samurai::cell_length(1), 0.5);
        EXPECT_DOUBLE_EQ(samurai::cell_length(3), 0.125);
        EXPECT_DOUBLE_EQ(samurai::cell_length(20), 9.5367431640625e-07 );
    }

    /**
     * Test of cell::face_center function (cell.hpp)
    */
    TEST(cell, face_center){
        auto indices = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        samurai::Cell<2, Interval<int>> c { 1, indices, 0 };

        {
            auto dir_x_p = xt::xtensor_fixed<int, xt::xshape<2>>({1, 0});
            auto fxp = c.face_center( dir_x_p );
            EXPECT_DOUBLE_EQ(  fxp(0), 1. );
        }

        {
            auto dir_x_m = xt::xtensor_fixed<int, xt::xshape<2>>({-1, 0});
            auto fxp = c.face_center( dir_x_m );
            EXPECT_DOUBLE_EQ(  fxp(0), 0.5 );
        }

        {
            auto dir_y_p = xt::xtensor_fixed<int, xt::xshape<2>>({0, 1});
            auto fxp = c.face_center( dir_y_p );
            EXPECT_DOUBLE_EQ(  fxp(1), 1. );
        }

    }

}