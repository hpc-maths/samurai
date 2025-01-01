#include <cstddef>
#include <xtensor/xfixed.hpp>

#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/interval.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/subset/apply.hpp>
#include <samurai/subset/interval_interface.hpp>
#include <samurai/subset/node.hpp>

namespace samurai::experimental
{
    TEST(new_subset, compute_min)
    {
        EXPECT_EQ(1, compute_min(3, 4, 1, 4));
        EXPECT_EQ(0, compute_min(0, 0, 0, 0));
        EXPECT_EQ(-1, compute_min(-1, -1, -1, -1));
    }

    TEST(new_subset, check_dim)
    {
        samurai::LevelCellArray<1> lca_1d;
        auto set_1d = self(lca_1d);
        static_assert(decltype(set_1d)::dim == 1);
        static_assert(decltype(intersection(set_1d, set_1d))::dim == 1);

        samurai::LevelCellArray<2> lca_2d;
        auto set_2d = self(lca_2d);
        static_assert(decltype(set_2d)::dim == 2);
        static_assert(decltype(intersection(set_2d, set_2d))::dim == 2);
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
                  [](auto& i, auto)
                  {
                      EXPECT_EQ(interval_t(0, 2), i);
                  });
        }

        {
            auto set = intersection(self(ca[5]).on(1), ca[1]);
            EXPECT_EQ(set.level(), 1);
            apply(set,
                  [](auto& i, auto)
                  {
                      EXPECT_EQ(interval_t(0, 2), i);
                  });

            EXPECT_EQ(set.on(3).level(), 3);
            apply(set.on(3),
                  [](auto& i, auto)
                  {
                      EXPECT_EQ(interval_t(0, 8), i);
                  });
        }

        {
            auto set = union_(ca[4], intersection(self(ca[5]).on(1), ca[1])).on(4);
            EXPECT_EQ(set.level(), 4);
            apply(set,
                  [](auto& i, auto)
                  {
                      EXPECT_EQ(interval_t(0, 20), i);
                  });

            EXPECT_EQ(set.on(5).level(), 5);
            apply(set,
                  [](auto& i, auto)
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
              [&never_call](auto&, auto)
              {
                  never_call = false;
              });
        EXPECT_TRUE(never_call);

        apply(intersection(translate(A, xt::xtensor_fixed<int, xt::xshape<1>>{2}), B).on(4),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(6, 8), i);
              });

        apply(translate(intersection(A, B).on(1), xt::xtensor_fixed<int, xt::xshape<1>>{5}).on(4),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(40, 48), i);
              });

        apply(translate(A, xt::xtensor_fixed<int, xt::xshape<1>>{2}).on(2),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(1, 2), i);
              });

        apply(translate(B, xt::xtensor_fixed<int, xt::xshape<1>>{-2}).on(2),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(0, 2), i);
              });

        apply(translate(B, xt::xtensor_fixed<int, xt::xshape<1>>{-4}).on(1).on(2),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(-2, 2), i);
              });

        apply(translate(B, xt::xtensor_fixed<int, xt::xshape<1>>{-2}).on(4),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(2, 6), i);
              });

        never_call = true;
        apply(intersection(intersection(self(A).on(2), self(B).on(2)).on(1), C),
              [&never_call](auto&, auto)
              {
                  never_call = false;
              });
        EXPECT_TRUE(never_call);

        apply(intersection(self(A).on(1), self(B).on(1)).on(2).on(1).on(3),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(0, 4), i);
              });

        apply(intersection(self(A).on(1), B).on(2),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(1, 2), i);
              });

        apply(intersection(union_(self(A).on(2), self(B).on(2)), B).on(2),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(1, 3), i);
              });
        apply(self(A).on(1).on(2),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(0, 2), i);
              });

        apply(self(A).on(2).on(1),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(0, 1), i);
              });

        apply(translate(self(A).on(2).on(1), xt::xtensor_fixed<int, xt::xshape<1>>{1}),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(1, 2), i);
              });
    }

    // TEST(new_subset, offset_iterator)
    // {
    //     using interval_t          = Interval<int>;
    //     std::vector<interval_t> x = {
    //         {0, 2},
    //         {4, 5},
    //         {0, 1},
    //         {2, 4},
    //         {7, 9}
    //     };
    //     std::vector<std::size_t> offset = {0, 2, 4, 5};

    //     auto begin = offset_iterator(x.cbegin(), x.cbegin() + 1);
    //     auto end   = offset_iterator(x.cbegin() + 1, x.cbegin() + 1);
    //     auto it    = begin;
    //     EXPECT_EQ(interval_t(0, 2), *it);
    //     it++;
    //     EXPECT_EQ(interval_t(4, 5), *it);
    //     it++;
    //     it++;
    //     EXPECT_TRUE(it == end);

    //     std::vector<interval_t> y = {
    //         {2, 4},
    //         {3, 5},
    //         {4, 6}
    //     };
    //     using iterator_t = decltype(y.cbegin());
    //     std::vector<iterator_t> obegin{y.cbegin(), y.cbegin() + 1, y.cbegin() + 2};
    //     std::vector<iterator_t> oend{y.cbegin() + 1, y.cbegin() + 2, y.cbegin() + 3};
    //     auto ybegin = offset_iterator<iterator_t>(obegin, oend);
    //     auto yend   = offset_iterator<iterator_t>(oend, oend);
    //     auto yit    = ybegin;
    //     EXPECT_EQ(interval_t(2, 6), *yit);
    //     yit++;
    //     yit++;
    //     EXPECT_TRUE(yit == yend);
    // }

    // TEST(new_subset, union_of_offset)
    // {
    //     using interval_t          = Interval<int>;
    //     int level                 = 0;
    //     std::vector<interval_t> x = {
    //         {0, 2},
    //         {4, 5},
    //         {0, 1},
    //         {3, 4},
    //         {7, 9}
    //     };
    //     std::vector<std::size_t> offset = {0, 2, 4, 5};

    //     using iterator_t = decltype(x.cbegin());
    //     std::vector<iterator_t> obegin{x.cbegin(), x.cbegin() + 2, x.cbegin() + 4};
    //     std::vector<iterator_t> oend{x.cbegin() + 2, x.cbegin() + 4, x.cbegin() + 5};
    //     auto begin = offset_iterator(obegin, oend);
    //     auto end   = offset_iterator(oend, oend);
    //     auto set_1 = IntervalVector(level + 2, level + 1, level + 1, level + 2, begin, end);

    //     apply(set_1,
    //           [](auto& i)
    //           {
    //               EXPECT_EQ(interval_t(0, 5), i);
    //           });

    //     obegin.erase(obegin.begin());
    //     oend.erase(oend.begin());
    //     begin = offset_iterator(obegin, oend);
    //     end   = offset_iterator(oend, oend);

    //     auto set_2 = IntervalVector(level + 2, level, level, level + 2, begin, end);

    //     apply(set_2,
    //           [](auto& i)
    //           {
    //               EXPECT_EQ(interval_t(0, 3), i);
    //           });
    // }

    TEST(new_subset, one_interval)
    {
        samurai::CellList<1> cl;
        samurai::CellArray<1> ca;

        cl[4][{}].add_interval({2, 4});
        cl[4][{}].add_interval({5, 6});

        ca = {cl, true};

        auto func = [](int, int i)
        {
            return i;
        };

        auto set = IntervalVector(4, 4, 4, ca[4][0].begin(), ca[4][0].begin() + 1, func, func);
        apply(set,
              [](auto& i)
              {
                  std::cout << i << std::endl;
              });
    }

    // TEST(new_subset, 2d_case)
    // {
    //     samurai::CellList<2> cl;
    //     samurai::CellArray<2> ca1, ca2;

    //     cl[4][{-1}].add_interval({2, 4});
    //     cl[4][{0}].add_interval({3, 5});
    //     cl[4][{1}].add_interval({4, 6});

    //     ca1 = {cl, true};

    //     cl.clear();
    //     cl[5][{-1}].add_interval({5, 7});
    //     cl[5][{0}].add_interval({3, 5});
    //     cl[5][{1}].add_interval({4, 6});
    //     ca2 = {cl, true};

    //     apply(self(ca1[4]).on(3),
    //           [](auto& i, auto index)
    //           {
    //               std::cout << "solution: " << i << " " << index[0] << std::endl;
    //           });

    //     // apply(intersection(ca1[4], ca2[4]),
    //     //       [](auto& i, auto index)
    //     //       {
    //     //           std::cout << i << " " << index[0] << std::endl;
    //     //       });
    // }

    TEST(new_subset, translate)
    {
        samurai::CellList<1> cl;
        samurai::CellArray<1> ca;

        cl[14][{}].add_interval({8612, 8620});
        cl[13][{}].add_interval({4279, 4325});

        ca = {cl, true};

        apply(translate(intersection(translate(ca[14], xt::xtensor_fixed<int, xt::xshape<1>>{-1}), self(ca[13]).on(14)),
                        xt::xtensor_fixed<int, xt::xshape<1>>{-2}),
              [](auto& i, auto)
              {
                  std::cout << "solution: " << i << std::endl;
              });
    }

    TEST(new_subset, translate_test)
    {
        samurai::CellList<1> cl;
        samurai::CellArray<1> ca;
        using interval_t = typename samurai::CellArray<1>::interval_t;

        cl[5][{}].add_interval({3, 17});

        ca = {cl, true};

        apply(intersection(translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{-2}), ca[5]),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(3, 15), i);
              });

        apply(intersection(translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{-2}).on(4), ca[5]),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(3, 16), i);
              });

        apply(intersection(translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{-2}).on(4),
                           translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{2})),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(5, 16), i);
              });

        apply(translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{-2}).on(4),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(0, 8), i);
              });

        apply(translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{2}).on(3),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(1, 5), i);
              });

        apply(intersection(translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{-2}).on(4),
                           translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{2}).on(3)),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(2, 8), i);
              });

        apply(translate(intersection(translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{-2}).on(4),
                                     translate(ca[5], xt::xtensor_fixed<int, xt::xshape<1>>{2}).on(3)),
                        xt::xtensor_fixed<int, xt::xshape<1>>{5}),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(7, 13), i);
              });
    }
}