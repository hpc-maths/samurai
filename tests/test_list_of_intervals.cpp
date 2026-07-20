#include <gtest/gtest.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

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

    // The following tests exercise the search hint cached on the last interval
    // touched, which makes the insertion in increasing order O(1).

    TEST(list_of_intervals, add_intervals_in_several_increasing_passes)
    {
        // Typical of the building of a cell list: a first pass fills the level
        // from the refined cells of the level below, a second one restarts from
        // the beginning with the cells already at that level.
        ListOfIntervals<int, int> list;
        for (int i = 0; i < 100; i += 4)
        {
            list.add_interval({i, i + 1});
        }
        for (int i = 2; i < 100; i += 4)
        {
            list.add_interval({i, i + 1});
        }
        // and a decreasing pass, to make sure the hint is never trusted blindly
        for (int i = 96; i >= 0; i -= 4)
        {
            list.add_interval({i + 1, i + 2});
        }

        EXPECT_EQ(list.size(), 25u);
        int expected_start = 0;
        for (const auto& interval : list)
        {
            EXPECT_EQ(interval.start, expected_start);
            EXPECT_EQ(interval.end, expected_start + 3);
            expected_start += 4;
        }
    }

    TEST(list_of_intervals, clear_then_add)
    {
        ListOfIntervals<int, int> list;
        list.add_interval({-3, 3});
        list.clear();
        EXPECT_EQ(list.size(), 0u);

        list.add_point(7);
        list.add_point(9);
        list.add_point(8);
        xt::xarray<Interval<int, int>> expected{
            {7, 10}
        };
        EXPECT_EQ(list, expected);
    }

    TEST(list_of_intervals, copy_then_add)
    {
        ListOfIntervals<int, int> list;
        list.add_interval({-3, 3});

        ListOfIntervals<int, int> copy_constructed(list);
        copy_constructed.add_interval({10, 12});

        ListOfIntervals<int, int> copy_assigned;
        copy_assigned.add_interval({-100, -50});
        copy_assigned = list;
        copy_assigned.add_interval({10, 12});

        xt::xarray<Interval<int, int>> expected{
            {-3, 3 },
            {10, 12}
        };
        EXPECT_EQ(copy_constructed, expected);
        EXPECT_EQ(copy_assigned, expected);
    }

    TEST(list_of_intervals, move_then_add)
    {
        xt::xarray<Interval<int, int>> expected{
            {-3, 3 },
            {10, 12}
        };

        ListOfIntervals<int, int> list;
        list.add_interval({-3, 3});
        ListOfIntervals<int, int> move_constructed(std::move(list));
        move_constructed.add_interval({10, 12});
        EXPECT_EQ(move_constructed, expected);

        ListOfIntervals<int, int> other;
        other.add_interval({-3, 3});
        ListOfIntervals<int, int> move_assigned;
        move_assigned.add_interval({-100, -50});
        move_assigned = std::move(other);
        move_assigned.add_interval({10, 12});
        EXPECT_EQ(move_assigned, expected);
    }

    TEST(list_of_intervals, add_interval_swallowing_the_last_one)
    {
        ListOfIntervals<int, int> list;
        list.add_interval({-3, -1});
        list.add_interval({2, 4});
        list.add_interval({8, 10});

        // swallows every interval, including the last one
        list.add_interval({-5, 12});
        xt::xarray<Interval<int, int>> expected{
            {-5, 12}
        };
        EXPECT_EQ(list, expected);

        // the cached iterator must still designate the merged interval
        list.add_interval({12, 14});
        xt::xarray<Interval<int, int>> expected_after{
            {-5, 14}
        };
        EXPECT_EQ(list, expected_after);
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
