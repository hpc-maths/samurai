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

    TEST(interval, right_shift)
    {
        using interval_t = Interval<int, int>;
        interval_t out;

        interval_t i{0, 9};

        out = i >> 1;
        EXPECT_EQ(out.start, 0);
        EXPECT_EQ(out.end, 5);

        out = i >> 2;
        EXPECT_EQ(out.start, 0);
        EXPECT_EQ(out.end, 3);

        out = i >> 20;
        EXPECT_EQ(out.start, 0);
        EXPECT_EQ(out.end, 1);

        i   = {-9, 2};
        out = i >> 1;
        EXPECT_EQ(out.start, -5);
        EXPECT_EQ(out.end, 1);

        out = i >> 2;
        EXPECT_EQ(out.start, -3);
        EXPECT_EQ(out.end, 1);

        out = i >> 10;
        EXPECT_EQ(out.start, -1);
        EXPECT_EQ(out.end, 1);
    }

    TEST(interval, ostream)
    {
        Interval<int, int> i{0, 3, 0};
        std::stringstream ss;
        ss << i;
        EXPECT_STREQ(ss.str().data(), "[0,3[@0:1");
    }
}
