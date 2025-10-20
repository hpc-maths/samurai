#include <map>

#include <gtest/gtest.h>

#include <xtensor/containers/xarray.hpp>

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
        // Expected mapping of an interval's start when applying a right shift of 1 (to be checked manually)
        const std::map<int, int> start_rshift{
            {-6, -3},
            {-5, -3},
            {-4, -2},
            {-3, -2},
            {-2, -1},
            {-1, -1},
            {0,  0 },
            {1,  0 },
            {2,  1 },
            {3,  1 },
            {4,  2 },
            {5,  2 },
            {6,  3 }
        };

        // Expected mapping of an interval's end when applying a right shift of 1 (to be checked manually)
        const std::map<int, int> end_rshift{
            {-6, -3},
            {-5, -2},
            {-4, -2},
            {-3, -1},
            {-2, -1},
            {-1, 0 },
            {0,  0 },
            {1,  1 },
            {2,  1 },
            {3,  2 },
            {4,  2 },
            {5,  3 },
            {6,  3 }
        };

        using interval_t = Interval<int, int>;

        // Greedy check of all possible intervals within [-6, 6]
        for (int start = -6; start < 6; ++start)
        {
            for (int end = start + 1; end <= 6; ++end)
            {
                interval_t i{start, end};
                for (std::size_t shift : {0u, 1u, 2u, 10u})
                {
                    // Computing resulting start & end (shifting by n is equivalent to n shift of 1)
                    int rstart = start;
                    for (std::size_t s = 0; s < shift; ++s)
                    {
                        rstart = start_rshift.at(rstart);
                    }
                    int rend = end;
                    for (std::size_t s = 0; s < shift; ++s)
                    {
                        rend = end_rshift.at(rend);
                    }

                    // Checking shifted interval
                    interval_t out = i >> shift;
                    EXPECT_EQ(out.start, rstart) << " for [" << start << "," << end << "[ >> " << shift;
                    EXPECT_EQ(out.end, rend) << " for [" << start << "," << end << "[ >> " << shift;
                }
            }
        }

        // Keeping old tests (in case of...)
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

    TEST(interval, left_shift)
    {
        // The expected mapping of an interval's start & end when applying a left shift of s
        // is simply a ---> a * 2^s = a << s

        using interval_t = Interval<int, int>;

        // Greedy check of all possible intervals within [-6, 6]
        for (int start = -6; start < 6; ++start)
        {
            for (int end = start + 1; end <= 6; ++end)
            {
                interval_t i{start, end};
                for (std::size_t shift : {0u, 1u, 2u, 10u})
                {
                    // Computing resulting start & end (simply `a << shift` for left shift)
                    int rstart = start << shift;
                    int rend   = end << shift;

                    // Checking shifted interval
                    interval_t out = i << shift;
                    EXPECT_EQ(out.start, rstart) << " for [" << start << "," << end << "[ << " << shift;
                    EXPECT_EQ(out.end, rend) << " for [" << start << "," << end << "[ << " << shift;
                }
            }
        }
    }

    TEST(interval, ostream)
    {
        Interval<int, int> i{0, 3, 0};
        std::stringstream ss;
        ss << i;
        EXPECT_STREQ(ss.str().data(), "[0,3[@0:1");
    }
}
