#include <cstddef>
#include <span>
#include <tuple>
#include <xtensor/xfixed.hpp>

#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/interval.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/subset/apply.hpp>
#include <samurai/subset/interval_interface.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    TEST(new_subset, lower_bound)
    {
        LevelCellList<1> lcl{1};
        LevelCellArray<1> lca;
        using interval_t = typename LevelCellArray<1>::interval_t;

        lcl[{}].add_interval({0, 2});
        lcl[{}].add_interval({9, 12});
        lcl[{}].add_interval({14, 20});

        lca = lcl;

        auto it = lower_bound_interval(lca[0].begin(), lca[0].end(), -1);
        EXPECT_EQ(it - lca[0].begin(), 0);

        it = lower_bound_interval(lca[0].begin(), lca[0].end(), 1);
        EXPECT_EQ(it - lca[0].begin(), 0);

        it = lower_bound_interval(lca[0].begin(), lca[0].end(), 5);
        EXPECT_EQ(it - lca[0].begin(), 1);

        it = lower_bound_interval(lca[0].begin(), lca[0].end(), 12);
        EXPECT_EQ(it - lca[0].begin(), 2);

        it = lower_bound_interval(lca[0].begin(), lca[0].end(), 20);
        EXPECT_TRUE(it == lca[0].end());

        it = lower_bound_interval(lca[0].begin(), lca[0].end(), 25);
        EXPECT_TRUE(it == lca[0].end());
    }

    TEST(new_subset, upper_bound)
    {
        LevelCellList<1> lcl{1};
        LevelCellArray<1> lca;
        using interval_t = typename LevelCellArray<1>::interval_t;

        lcl[{}].add_interval({0, 2});
        lcl[{}].add_interval({9, 12});
        lcl[{}].add_interval({14, 20});

        lca = lcl;

        auto it = upper_bound_interval(lca[0].begin(), lca[0].end(), -1);
        EXPECT_EQ(it - lca[0].begin(), 0);

        it = upper_bound_interval(lca[0].begin(), lca[0].end(), 1);
        EXPECT_EQ(it - lca[0].begin(), 1);

        it = upper_bound_interval(lca[0].begin(), lca[0].end(), 5);
        EXPECT_EQ(it - lca[0].begin(), 1);

        it = upper_bound_interval(lca[0].begin(), lca[0].end(), 12);
        EXPECT_EQ(it - lca[0].begin(), 2);

        it = upper_bound_interval(lca[0].begin(), lca[0].end(), 20);
        EXPECT_TRUE(it == lca[0].end());

        it = upper_bound_interval(lca[0].begin(), lca[0].end(), 25);
        EXPECT_TRUE(it == lca[0].end());
    }

    TEST(utils, Self)
    {
        using interval_t = typename LevelCellArray<2>::interval_t;
        using expected_t = std::vector<std::pair<int, interval_t>>;

        LevelCellList<2> lcl{1};
        LevelCellArray<2> lca;
        lcl[{1}].add_interval({0, 2});
        lcl[{1}].add_interval({9, 12});
        lcl[{1}].add_interval({14, 20});
        lcl[{2}].add_interval({1, 4});
        lcl[{2}].add_interval({10, 20});

        lca = lcl;

        auto expected = expected_t{
            {0, {0, 1} },
            {0, {4, 6} },
            {0, {7, 10}},
            {1, {0, 2} },
            {1, {5, 10}}
        };
        std::size_t ie = 0;
        self(lca).on(0)(
            [&](auto& i, auto& index)
            {
                EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
            });
    }

    TEST(utils, Self1d)
    {
        using interval_t = typename LevelCellArray<1>::interval_t;
        using expected_t = std::vector<interval_t>;

        LevelCellList<1> lcl{3};
        LevelCellArray<1> lca;
        lcl[{}].add_interval({0, 2});
        lcl[{}].add_interval({9, 12});

        lca = lcl;

        expected_t expected{
            {0, 2 },
            {9, 12}
        };
        std::size_t ie = 0;
        self(lca)(
            [&](auto& i, auto)
            {
                EXPECT_EQ(expected[ie++], i);
            });

        expected.clear();
        expected = {
            {0, 1},
            {2, 3}
        };
        ie = 0;
        self(lca).on(1)(
            [&](auto& i, auto)
            {
                EXPECT_EQ(expected[ie++], i);
            });
    }

    TEST(new_subset, compute_min)
    {
        EXPECT_EQ(1, compute_min(3, 4, 1, 4));
        EXPECT_EQ(0, compute_min(0, 0, 0, 0));
        EXPECT_EQ(-1, compute_min(-1, -1, -1, -1));
    }

    TEST(new_subset, check_dim)
    {
        LevelCellArray<1> lca_1d;
        auto set_1d = self(lca_1d);
        static_assert(decltype(set_1d)::dim == 1);
        static_assert(decltype(intersection(set_1d, set_1d))::dim == 1);

        LevelCellArray<2> lca_2d;
        auto set_2d = self(lca_2d);
        static_assert(decltype(set_2d)::dim == 2);
        static_assert(decltype(intersection(set_2d, set_2d))::dim == 2);
    }

    TEST(new_subset, test1)
    {
        CellList<1> cl;
        CellArray<1> ca;
        using interval_t = typename CellArray<1>::interval_t;

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

        LevelCellList<1> Al(3);
        Al[{}].add_point(0);
        LevelCellArray<1> A{Al};

        LevelCellList<1> Bl(3);
        Bl[{}].add_interval({3, 5});
        LevelCellArray<1> B{Bl};

        LevelCellList<1> Cl(1);
        Cl[{}].add_point(1);
        LevelCellArray<1> C{Cl};

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

    TEST(new_subset, 2d_case)
    {
        CellList<2> cl;
        CellArray<2> ca1, ca2;
        using interval_t = typename CellArray<2>::interval_t;
        using expected_t = std::vector<std::pair<int, interval_t>>;

        cl[4][{-1}].add_interval({2, 4});
        cl[4][{0}].add_interval({3, 5});
        cl[4][{1}].add_interval({4, 6});

        ca1 = {cl, true};

        cl.clear();
        cl[5][{-1}].add_interval({5, 7});
        cl[5][{0}].add_interval({3, 5});
        cl[5][{1}].add_interval({4, 6});
        ca2 = {cl, true};

        {
            auto expected = expected_t{
                {-1, {0, 1}},
                {0,  {0, 2}}
            };
            std::size_t ie = 0;
            apply(self(ca1[4]).on(2),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });
        }

        {
            bool never_call = true;
            apply(intersection(ca1[4], ca2[4]),
                  [&never_call](auto&, auto)
                  {
                      never_call = false;
                  });
            EXPECT_TRUE(never_call);
        }

        {
            auto expected = expected_t{
                {-1, {5, 7}},
                {0,  {3, 5}},
                {1,  {4, 6}}
            };
            std::size_t ie = 0;
            apply(intersection(ca1[4], ca2[5]),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });
        }

        {
            auto expected = expected_t{
                {0, {6, 7}}
            };
            std::size_t ie = 0;
            apply(intersection(ca1[4], translate(ca2[5], xt::xtensor_fixed<int, xt::xshape<2>>{0, 1})),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });
        }

        {
            auto expected = expected_t{
                {-1, {2, 4}},
                {0,  {3, 5}},
                {1,  {4, 6}}
            };
            std::size_t ie = 0;
            apply(union_(ca1[4], ca2[4]),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });
        }

        {
            cl.clear();
            cl[8][{32}].add_interval({36, 64});
            cl[8][{32}].add_interval({88, 116});
            cl[8][{33}].add_interval({36, 64});
            cl[8][{33}].add_interval({88, 116});

            CellArray<2> ca = {cl, true};

            auto expected = expected_t{
                {16, {18, 32}},
                {16, {44, 58}}
            };
            std::size_t ie = 0;
            apply(union_(ca[8], ca[7]).on(7),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });
        }

        {
            cl.clear();
            cl[7][{0}].add_interval({0, 128});
            CellArray<2> ca_1 = {cl, true};
            cl.clear();
            cl[7][{10}].add_interval({10, 66});
            CellArray<2> ca_2 = {cl, true};

            auto expected = expected_t{
                {0, {0, 64}},
                {5, {5, 33}}
            };
            std::size_t ie = 0;
            apply(union_(ca_1[7], ca_2[7]).on(6),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });
        }

        {
            cl.clear();
            cl[7][{0}].add_interval({0, 128});
            CellArray<2> ca_1 = {cl, true};

            auto expected = expected_t{
                {-1, {0, 2}}
            };
            std::size_t ie = 0;

            apply(translate(translate(self(ca_1[7]).on(5), xt::xtensor_fixed<int, xt::xshape<2>>{0, 2}).on(3),
                            xt::xtensor_fixed<int, xt::xshape<2>>{0, -1})
                      .on(1),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(ie, 0);
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });
        }

        {
            using Config = MRConfig<2>;
            const Box<double, 2> box({0, 0}, {1, 1});
            MRMesh<Config> mesh{box, 0, 3};
            auto& domain                              = mesh.domain();
            xt::xtensor_fixed<int, xt::xshape<2>> dir = {0, 1 << (3 - 1)};

            auto expected = expected_t{
                {0, {0, 2}}
            };
            std::size_t ie = 0;

            apply(difference(domain, translate(domain, dir)).on(1),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(ie, 0);
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });

            dir = {0, 1};
            ie  = 0;
            apply(difference(self(domain).on(1), translate(self(domain).on(1), dir)),
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(ie, 0);
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                  });
        }
    }

    TEST(new_subset, translate)
    {
        CellList<1> cl;
        CellArray<1> ca;
        using interval_t = typename CellArray<1>::interval_t;

        cl[14][{}].add_interval({8612, 8620});
        cl[13][{}].add_interval({4279, 4325});

        ca = {cl, true};

        apply(translate(intersection(translate(ca[14], xt::xtensor_fixed<int, xt::xshape<1>>{-1}), self(ca[13]).on(14)),
                        xt::xtensor_fixed<int, xt::xshape<1>>{-2}),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(8609, 8617), i);
              });
    }

    TEST(new_subset, translate_test)
    {
        CellList<1> cl;
        CellArray<1> ca;
        using interval_t = typename CellArray<1>::interval_t;

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