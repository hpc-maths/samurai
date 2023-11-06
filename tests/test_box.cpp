#include <algorithm>

#include <gtest/gtest.h>
// #include <rapidcheck/gtest.h>

#include <xtensor/xarray.hpp>

#include <samurai/box.hpp>

namespace samurai
{

    // RC_GTEST_PROP(Box, corner,
    //               (std::array<int, 2> min,
    //                std::array<int, 2> max))
    // {
    //     RC_PRE(min[0] < max[0]);
    //     RC_PRE(min[1] < max[1]);

    //     Box<int, 2> box{{min[0], min[1]}, {max[0], max[1]}};
    //     xt::xarray<int> expected{min[0], min[1]};
    //     RC_ASSERT(box.min_corner() == expected);
    // }

    TEST(box, min_corner)
    {
        Box<int, 2> box{
            {0, 0},
            {1, 1}
        };
        xt::xarray<int> expected{0, 0};
        EXPECT_EQ(box.min_corner(), expected);
    }

    TEST(box, max_corner)
    {
        Box<int, 2> box{
            {0, 0},
            {1, 1}
        };
        xt::xarray<int> expected{1, 1};
        EXPECT_EQ(box.max_corner(), expected);
    }

    TEST(box, length)
    {
        Box<int, 2> box_1{
            {0,  0},
            {10, 5}
        };
        xt::xarray<int> expected_1{10, 5};

        Box<int, 2> box_2{
            {-5, -10},
            {10, 5  }
        };
        xt::xarray<int> expected_2{15, 15};

        EXPECT_EQ(box_1.length(), expected_1);
        EXPECT_EQ(box_2.length(), expected_2);
    }

    TEST(box, is_valid)
    {
        EXPECT_TRUE((Box<int, 2>{
                         {0, 0},
                         {1, 1}
        })
                        .is_valid());
        EXPECT_FALSE((Box<int, 2>{
                          {1, 1},
                          {0, 0}
        })
                         .is_valid());
        EXPECT_FALSE((Box<int, 2>{
                          {0, 0},
                          {1, 0}
        })
                         .is_valid());
        EXPECT_FALSE((Box<int, 2>{
                          {0, 0},
                          {0, 1}
        })
                         .is_valid());
    }

    TEST(box, operator)
    {
        Box<int, 2> b{
            {-1, -1},
            {1,  1 }
        };
        xt::xarray<int> expected_min{-5, -5};
        xt::xarray<int> expected_max{5, 5};

        auto bl = b * 5;
        auto br = 5 * b;
        EXPECT_EQ(bl.min_corner(), expected_min);
        EXPECT_EQ(br.min_corner(), expected_min);
        EXPECT_EQ(bl.max_corner(), expected_max);
        EXPECT_EQ(br.max_corner(), expected_max);
        EXPECT_EQ(bl.length(), 5 * b.length());
        EXPECT_EQ(br.length(), 5 * b.length());
    }

    TEST(box, ostream)
    {
        Box<int, 2> b{
            {-1, -1},
            {1,  1 }
        };
        std::stringstream ss;
        ss << b;
        EXPECT_STREQ(ss.str().data(), "Box({-1, -1}, {1, 1})");
    }

}
