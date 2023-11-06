#include <gtest/gtest.h>

#include <xtensor/xarray.hpp>

#include <samurai/interval.hpp>

#include "test_common.hpp"

namespace samurai
{
    TEST(interval, size)
    {
        Interval<int, int> i{0, 3, 0};
        EXPECT_EQ(i.size(), 3U);
    }

    TEST(interval, contains)
    {
        Interval<int, int> i{0, 3, 0};
        EXPECT_TRUE(i.contains(2));
        EXPECT_FALSE(i.contains(-1));
    }

    TEST(interval, is_valid)
    {
        Interval<int, int> i1{0, 3, 0};
        Interval<int, int> i2{0, -3, 0};
        EXPECT_TRUE(i1.is_valid());
        EXPECT_FALSE(i2.is_valid());
    }

    TEST(interval, operator)
    {
        Interval<int, int> i{0, 3, 0};

        auto i1 = 3 * i;
        EXPECT_EQ(i1.start, 0);
        EXPECT_EQ(i1.end, 9);
        EXPECT_EQ(i1.index, 0);
        EXPECT_EQ(i1.step, 3);

        i1 = i * 3;
        EXPECT_EQ(i1.start, 0);
        EXPECT_EQ(i1.end, 9);
        EXPECT_EQ(i1.index, 0);
        EXPECT_EQ(i1.step, 3);

        i1 = 3 + i;
        EXPECT_EQ(i1.start, 3);
        EXPECT_EQ(i1.end, 6);
        EXPECT_EQ(i1.index, 0);
        EXPECT_EQ(i1.step, 1);

        i1 = i + 3;
        EXPECT_EQ(i1.start, 3);
        EXPECT_EQ(i1.end, 6);
        EXPECT_EQ(i1.index, 0);
        EXPECT_EQ(i1.step, 1);

        i1 = 3 - i;
        EXPECT_EQ(i1.start, -3);
        EXPECT_EQ(i1.end, 0);
        EXPECT_EQ(i1.index, 0);
        EXPECT_EQ(i1.step, 1);

        i1 = i - 3;
        EXPECT_EQ(i1.start, -3);
        EXPECT_EQ(i1.end, 0);
        EXPECT_EQ(i1.index, 0);
        EXPECT_EQ(i1.step, 1);
    }

    TEST(interval, ostream)
    {
        Interval<int, int> i{0, 3, 0};
        std::stringstream ss;
        ss << i;
        EXPECT_STREQ(ss.str().data(), "[0,3[@0:1");
    }
}
