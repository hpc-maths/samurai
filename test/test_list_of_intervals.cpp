#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/interval.hpp>
#include <samurai/list_of_intervals.hpp>

#include "test_common.hpp"

namespace samurai
{
    TEST(list_of_intervals, add_interval)
    {
        ListOfIntervals<int, int> list;
        xt::xarray<Interval<int, int>> expected{
            {-3, 3}
        };
        list.add_interval({-3, 3});
        EXPECT_EQ(list, expected);
    }

    TEST(list_of_intervals, add_interval_back)
    {
        ListOfIntervals<int, int> list;
        xt::xarray<Interval<int, int>> expected{
            {-10, 3}
        };
        list.add_interval({-3, 3});
        list.add_interval({-10, -3});
        EXPECT_EQ(list, expected);
    }

    TEST(list_of_intervals, add_interval_front)
    {
        ListOfIntervals<int, int> list;
        xt::xarray<Interval<int, int>> expected{
            {-3, 10}
        };
        list.add_interval({-3, 3});
        list.add_interval({3, 10});
        EXPECT_EQ(list, expected);
    }

    TEST(list_of_intervals, add_interval_intersection_1)
    {
        ListOfIntervals<int, int> list;
        xt::xarray<Interval<int, int>> expected{
            {-10, 3}
        };
        list.add_interval({-3, 3});
        list.add_interval({-10, 0});
        EXPECT_EQ(list, expected);
    }

    TEST(list_of_intervals, add_interval_intersection_2)
    {
        ListOfIntervals<int, int> list;
        xt::xarray<Interval<int, int>> expected{
            {-3, 10}
        };
        list.add_interval({-3, 3});
        list.add_interval({0, 10});
        EXPECT_EQ(list, expected);
    }

    TEST(list_of_intervals, add_points)
    {
        ListOfIntervals<int, int> list;

        int size  = 1000;
        auto perm = -size / 2 + xt::random::permutation<int>(size);
        for (auto p : perm)
        {
            list.add_point(p);
        }

        xt::xarray<Interval<int, int>> expected{
            {-size / 2, size / 2}
        };
        EXPECT_EQ(list, expected);
    }

    TEST(list_of_intervals, size)
    {
        ListOfIntervals<int, int> list;
        EXPECT_EQ(list.size(), 0u);

        list.add_point(3);
        EXPECT_EQ(list.size(), 1u);

        list.add_point(5);
        EXPECT_EQ(list.size(), 2u);

        list.add_point(4);
        EXPECT_EQ(list.size(), 1u);
    }

    TEST(list_of_intervals, ostream)
    {
        ListOfIntervals<int, int> list;
        list.add_point(2);
        list.add_interval({4, 7});
        std::stringstream ss;
        ss << list;
        EXPECT_STREQ(ss.str().data(), "[2,3[@0:1 [4,7[@0:1 ");
    }
}
