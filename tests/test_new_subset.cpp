#include <cstddef>
#include <span>

#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/interval.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/subset_new/apply.hpp>
#include <samurai/subset_new/interval_interface.hpp>
#include <samurai/subset_new/node.hpp>

namespace samurai::experimental
{
    TEST(new_subset, compute_min)
    {
        EXPECT_EQ(1, compute_min(3, 4, 1, 4));
        EXPECT_EQ(0, compute_min(0, 0, 0, 0));
        EXPECT_EQ(-1, compute_min(-1, -1, -1, -1));
    }

    TEST(new_subset, test1)
    {
        samurai::CellList<1> cl;
        samurai::CellArray<1> ca;
        using interval_t = typename samurai::CellArray<1>::interval_t;

        cl[1][{}].add_point(0);
        cl[1][{}].add_point(1);

        cl[4][{}].add_interval({0, 2});
        cl[4][{}].add_interval({9, 12});
        cl[4][{}].add_interval({14, 20});

        cl[5][{}].add_interval({9, 12});
        cl[5][{}].add_interval({14, 20});

        ca = {cl, true};

        {
            auto set = intersection(self(ca[4]).on(0), ca[5]).on(1);
            apply(set,
                  [](auto& i)
                  {
                      EXPECT_EQ(interval_t(0, 2), i);
                  });
        }

        {
            auto set = intersection(self(ca[5]).on(1), ca[1]);
            EXPECT_EQ(set.level(), 1);
            apply(set,
                  [](auto& i)
                  {
                      EXPECT_EQ(interval_t(0, 2), i);
                  });

            EXPECT_EQ(set.on(3).level(), 3);
            apply(set.on(3),
                  [](auto& i)
                  {
                      EXPECT_EQ(interval_t(0, 8), i);
                  });
        }

        {
            auto set = union_(ca[4], intersection(self(ca[5]).on(1), ca[1])).on(4);
            EXPECT_EQ(set.level(), 4);
            apply(set,
                  [](auto& i)
                  {
                      EXPECT_EQ(interval_t(0, 20), i);
                  });

            EXPECT_EQ(set.on(5).level(), 5);
            apply(set,
                  [](auto& i)
                  {
                      EXPECT_EQ(interval_t(0, 40), i);
                  });
        }

        samurai::LevelCellList<1> Al(3);
        Al[{}].add_point(0);
        samurai::LevelCellArray<1> A{Al};

        samurai::LevelCellList<1> Bl(3);
        Bl[{}].add_interval({3, 5});
        samurai::LevelCellArray<1> B{Bl};

        samurai::LevelCellList<1> Cl(1);
        Cl[{}].add_point(1);
        samurai::LevelCellArray<1> C{Cl};

        bool never_call = true;
        apply(intersection(intersection(self(A).on(1), self(B).on(1)).on(2), C),
              [&never_call](auto&)
              {
                  never_call = false;
              });
        EXPECT_TRUE(never_call);

        apply(intersection(translation(A, std::array<int, 1>{2}), B).on(4),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(6, 8), i);
              });

        apply(translation(intersection(A, B).on(1), std::array<int, 1>{5}).on(4),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(40, 49), i);
              });

        apply(translation(A, std::array<int, 1>{2}).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(2, 3), i);
              });

        apply(translation(B, std::array<int, 1>{-2}).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(-1, 1), i);
              });

        apply(translation(B, std::array<int, 1>{-2}).on(1).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(-2, 2), i);
              });

        apply(translation(B, std::array<int, 1>{-2}).on(4),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(4, 8), i);
              });

        never_call = true;
        apply(intersection(intersection(self(A).on(2), self(B).on(2)).on(1), C),
              [&never_call](auto&)
              {
                  never_call = false;
              });
        EXPECT_TRUE(never_call);

        apply(intersection(self(A).on(1), self(B).on(1)).on(2).on(1).on(3),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(0, 4), i);
              });

        apply(intersection(self(A).on(1), B).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(1, 2), i);
              });

        apply(intersection(union_(self(A).on(2), self(B).on(2)), B).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(1, 3), i);
              });
        apply(self(A).on(1).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(0, 2), i);
              });

        apply(self(A).on(2).on(1),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(0, 1), i);
              });

        apply(translation(self(A).on(2).on(1), std::array<int, 1>{1}),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(1, 2), i);
              });
    }

    TEST(new_subset, offset_iterator)
    {
        using interval_t          = Interval<int>;
        std::vector<interval_t> x = {
            {0, 2},
            {4, 5},
            {0, 1},
            {2, 4},
            {7, 9}
        };
        std::vector<std::size_t> offset = {0, 2, 4, 5};

        auto begin = offset_iterator(x.cbegin(), offset);
        auto end   = offset_iterator(x.cbegin() + static_cast<std::ptrdiff_t>(offset.back()), offset, true);
        auto it    = begin;
        EXPECT_EQ(interval_t(0, 5), *it);
        it++;
        EXPECT_EQ(interval_t(7, 9), *it);
        it++;
        it++;
        EXPECT_TRUE(it == end);
    }

    TEST(new_subset, union_of_offset)
    {
        using interval_t          = Interval<int>;
        int level                 = 0;
        std::vector<interval_t> x = {
            {0, 2},
            {4, 5},
            {0, 1},
            {3, 4},
            {7, 9}
        };
        std::vector<std::size_t> offset = {0, 2, 4, 5};

        auto set_1 = IntervalVectorOffset(level + 2, level + 1, level + 1, level + 2, x, offset);

        apply(set_1,
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(0, 5), i);
              });

        auto partial_offset_view = std::span(offset).subspan(1, 3);
        auto set_2               = IntervalVectorOffset(level + 2, level, level, level + 2, x, partial_offset_view);

        apply(set_2,
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(0, 3), i);
              });
    }

}