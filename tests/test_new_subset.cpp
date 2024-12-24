#include "samurai/level_cell_array.hpp"
#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
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

        // {
        //     auto set = intersection(identity(ca[4]).on(0), ca[5]).on(1);
        //     apply(set,
        //           [](auto& i)
        //           {
        //               EXPECT_EQ(interval_t(0, 2), i);
        //           });
        // }

        // {
        //     auto set = intersection(identity(ca[5]).on(1), ca[1]);
        //     EXPECT_EQ(set.level(), 1);
        //     apply(set,
        //           [](auto& i)
        //           {
        //               EXPECT_EQ(interval_t(0, 2), i);
        //           });

        //     EXPECT_EQ(set.on(3).level(), 3);
        //     apply(set.on(3),
        //           [](auto& i)
        //           {
        //               EXPECT_EQ(interval_t(0, 8), i);
        //           });
        // }

        // {
        //     auto set = union_(ca[4], intersection(identity(ca[5]).on(1), ca[1])).on(4);
        //     EXPECT_EQ(set.level(), 4);
        //     apply(set,
        //           [](auto& i)
        //           {
        //               EXPECT_EQ(interval_t(0, 20), i);
        //           });

        //     EXPECT_EQ(set.on(5).level(), 5);
        //     apply(set,
        //           [](auto& i)
        //           {
        //               EXPECT_EQ(interval_t(0, 40), i);
        //           });
        // }

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
        apply(intersection(intersection(identity(A).on(1), identity(B).on(1)).on(2), C),
              [&never_call](auto&)
              {
                  never_call = false;
              });
        EXPECT_TRUE(never_call);

        apply(translation(Identity(A), std::array<int, 1>{2}).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(2, 3), i);
              });

        apply(translation(Identity(B), std::array<int, 1>{-2}).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(-1, 1), i);
              });

        apply(translation(Identity(B), std::array<int, 1>{-2}).on(1).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(-2, 2), i);
              });

        apply(translation(Identity(B), std::array<int, 1>{-2}).on(4),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(4, 8), i);
              });

        never_call = true;
        apply(intersection(intersection(identity(A).on(2), identity(B).on(2)).on(1), C),
              [&never_call](auto&)
              {
                  never_call = false;
              });
        EXPECT_TRUE(never_call);

        apply(intersection(identity(A).on(1), identity(B).on(1)).on(2).on(1).on(3),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(0, 4), i);
              });

        apply(intersection(identity(A).on(1), identity(B)).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(1, 2), i);
              });

        apply(intersection(union_(identity(A).on(2), identity(B).on(2)), identity(B)).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(1, 3), i);
              });
        apply(identity(A).on(1).on(2),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(0, 2), i);
              });

        apply(identity(A).on(2).on(1),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(0, 1), i);
              });

        apply(translation(identity(A).on(2).on(1), std::array<int, 1>{1}),
              [](auto& i)
              {
                  EXPECT_EQ(interval_t(1, 2), i);
              });
    }
}