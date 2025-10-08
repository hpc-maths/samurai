#include <cstddef>
#include <filesystem>
#include <span>
#include <tuple>
#include <xtensor/xfixed.hpp>

#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/interval.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/subset/node.hpp>
#include <xtensor/xtensor_forward.hpp>

#include <fmt/ranges.h>
#include <samurai/io/hdf5.hpp>

namespace samurai
{
    TEST(subset, lower_bound)
    {
        LevelCellList<1> lcl{1};
        LevelCellArray<1> lca;

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

    TEST(subset, upper_bound)
    {
        LevelCellList<1> lcl{1};
        LevelCellArray<1> lca;

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

    TEST(subset, compute_min)
    {
        EXPECT_EQ(1, vmin(3, 4, 1, 4));
        EXPECT_EQ(0, vmin(0, 0, 0, 0));
        EXPECT_EQ(-1, vmin(-1, -1, -1, -1));
    }

    TEST(subset, check_dim)
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

    TEST(subset, test1)
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
            auto set = self(ca[4]).on(0);
            apply(set,
                  [](auto& i, auto)
                  {
                      EXPECT_EQ(interval_t(0, 2), i);
                  });
        }

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

            auto set2 = set.on(5);
            EXPECT_EQ(set2.level(), 5);
            apply(set2,
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

        apply(translate(B, xt::xtensor_fixed<int, xt::xshape<1>>{-4}).on(2).on(1),
              [](auto& i, auto)
              {
                  EXPECT_EQ(interval_t(-1, 1), i);
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

    TEST(subset, 2d_case)
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
            fmt::print("====================================================\n");
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

            // apply(difference(domain, translate(domain, dir)).on(1),
            //       [&](auto& i, auto& index)
            //       {
            //           EXPECT_EQ(ie, 0);
            //           EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
            //       });

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

    TEST(subset, expand)
    {
        using interval_t = typename LevelCellArray<2>::interval_t;
        using expected_t = std::vector<std::pair<int, interval_t>>;

        LevelCellArray<2> ca;

        ca.add_interval_back({0, 1}, {0});

        {
            const auto translated_ca = translate(ca, {3 + 1, 0});
            const auto joined_cas    = union_(ca, translated_ca);

            const auto set = expand(joined_cas, 3);

            expected_t expected{
                {-3, {-3, 8}},
                {-2, {-3, 8}},
                {-1, {-3, 8}},
                {0,  {-3, 8}},
                {1,  {-3, 8}},
                {2,  {-3, 8}},
                {3,  {-3, 8}}
            };

            bool is_set_empty = true;
            std::size_t ie    = 0;
            set(
                [&expected, &is_set_empty, &ie](const auto& x_interval, const auto& yz)
                {
                    is_set_empty = false;
                    EXPECT_EQ(expected[ie++], std::make_pair(yz[0], x_interval));
                });
            EXPECT_EQ(ie, expected.size());
            EXPECT_FALSE(is_set_empty);
        }

        {
            const auto translated_ca = translate(ca, {0, 3 + 1});
            const auto joined_cas    = union_(ca, translated_ca);

            const auto set = expand(joined_cas, 3);

            expected_t expected{
                {-3, {-3, 4}},
                {-2, {-3, 4}},
                {-1, {-3, 4}},
                {0,  {-3, 4}},
                {1,  {-3, 4}},
                {2,  {-3, 4}},
                {3,  {-3, 4}},
                {4,  {-3, 4}},
                {5,  {-3, 4}},
                {6,  {-3, 4}},
                {7,  {-3, 4}}
            };

            bool is_set_empty = true;
            std::size_t ie    = 0;
            set(
                [&expected, &is_set_empty, &ie](const auto& x_interval, const auto& yz)
                {
                    is_set_empty = false;
                    EXPECT_EQ(expected[ie++], std::make_pair(yz[0], x_interval));
                });
            EXPECT_EQ(ie, expected.size());
            EXPECT_FALSE(is_set_empty);
        }
        {
            const auto translated_ca = translate(ca, {3 + 1, 3 + 1});
            const auto joined_cas    = union_(ca, translated_ca);

            const auto set = expand(joined_cas, 3);

            expected_t expected{
                {-3, {-3, 4}},
                {-2, {-3, 4}},
                {-1, {-3, 4}},
                {0,  {-3, 4}},
                {1,  {-3, 8}},
                {2,  {-3, 8}},
                {3,  {-3, 8}},
                {4,  {1, 8} },
                {5,  {1, 8} },
                {6,  {1, 8} },
                {7,  {1, 8} }
            };

            bool is_set_empty = true;
            std::size_t ie    = 0;
            set(
                [&expected, &is_set_empty, &ie](const auto& x_interval, const auto& yz)
                {
                    is_set_empty = false;
                    EXPECT_EQ(expected[ie++], std::make_pair(yz[0], x_interval));
                });
            EXPECT_EQ(ie, expected.size());
            EXPECT_FALSE(is_set_empty);

            const auto lca_joined_cas = joined_cas.to_lca();
            const auto lca_set        = set.to_lca();
        }
    }

    TEST(subset, contract)
    {
        LevelCellArray<2> ca;

        ca.add_interval_back({0, 1}, {0});

        {
            const auto translated_ca = translate(ca, {3 + 1, 0});
            const auto joined_cas    = union_(ca, translated_ca);

            const auto set = contract(joined_cas, 1);

            bool is_set_empty = true;
            set(
                [&is_set_empty](const auto& x_interval, const auto& yz)
                {
                    fmt::print("x_interval = {} -- yz = {}", x_interval, yz[0]);
                    is_set_empty = false;
                });
            EXPECT_TRUE(is_set_empty);
            //~ EXPECT_TRUE(set.empty());
        }
    }

    TEST(subset, translate)
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

    TEST(subset, translate_test)
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

    TEST(subset, translate_2d)
    {
        using lca_t      = LevelCellArray<2>;
        using interval_t = typename lca_t::interval_t;
        LevelCellArray<2> boundary(5);

        LevelCellArray<2> domain1(5);
        for (int j = 4; j < 6; ++j)
        {
            domain1.add_interval_back({0, 32}, {j});
        }
        xt::xtensor_fixed<int, xt::xshape<2>> translation{1, 0};
        auto diff = difference(domain1, translate(self(domain1), -translation));

        bool found = false;
        diff.on(4)(
            [&](auto& i, auto& index)
            {
                found = true;
                EXPECT_EQ(interval_t(15, 16), i);
                EXPECT_EQ(index[0], 2);
            });
        EXPECT_TRUE(found);

        found = false;
        intersection(diff.on(4), diff.on(4))(
            [&](auto& i, auto& index)
            {
                found = true;
                EXPECT_EQ(interval_t(15, 16), i);
                EXPECT_EQ(index[0], 2);
            });
        EXPECT_TRUE(found);

        found = false;
        intersection(diff.on(4), diff.on(4))
            .on(3)(
                [&](auto& i, auto& index)
                {
                    found = true;
                    EXPECT_EQ(interval_t(7, 8), i);
                    EXPECT_EQ(index[0], 1);
                });
        EXPECT_TRUE(found);

        found = false;
        intersection(intersection(diff.on(4), diff.on(4)).on(3), intersection(diff.on(4), diff.on(4)).on(5))
            .on(3)(
                [&](auto& i, auto& index)
                {
                    found = true;
                    EXPECT_EQ(interval_t(7, 8), i);
                    EXPECT_EQ(index[0], 1);
                });
        EXPECT_TRUE(found);

        LevelCellArray<2> lca_4(4);
        lca_4.add_interval_back({15, 16}, {2});

        found = false;
        intersection(intersection(diff.on(4), diff.on(4)).on(3), lca_4)(
            [&](auto& i, auto& index)
            {
                found = true;
                EXPECT_EQ(interval_t(15, 16), i);
                EXPECT_EQ(index[0], 2);
            });
        EXPECT_TRUE(found);
    }

    TEST(subset, union)
    {
        using interval_t = typename CellArray<2>::interval_t;
        using expected_t = std::vector<std::pair<int, interval_t>>;
        LevelCellList<2> lcl1(1);
        LevelCellList<2> lcl2(1);
        LevelCellArray<2> lca1;
        LevelCellArray<2> lca2;

        lcl1[{0}].add_interval({0, 1});
        lcl2[{1}].add_interval({1, 2});

        lca1 = lcl1;
        lca2 = lcl2;

        {
            auto expected = expected_t{
                {0, {0, 1}},
                {1, {1, 2}}
            };

            auto set = union_(lca1, lca2);
            EXPECT_EQ(set.level(), 1);

            bool found     = false;
            std::size_t ie = 0;
            apply(set,
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                      found = true;
                  });
            EXPECT_TRUE(found);
        }

        {
            auto set = intersection(lca1, lca2);
            EXPECT_EQ(set.level(), 1);

            bool found = false;
            apply(set,
                  [&](auto&, auto&)
                  {
                      found = true;
                  });
            EXPECT_FALSE(found);
        }

        {
            auto expected = expected_t{
                {0, {0, 1}},
            };

            auto set = intersection(lca1, self(lca2).on(0));
            EXPECT_EQ(set.level(), 1);

            bool found     = false;
            std::size_t ie = 0;
            apply(set,
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                      found = true;
                  });
            EXPECT_TRUE(found);
        }

        {
            auto set = difference(lca1, self(lca2).on(0));
            EXPECT_EQ(set.level(), 1);

            bool found = false;
            apply(set,
                  [&](auto&, auto&)
                  {
                      found = true;
                  });
            EXPECT_FALSE(found);
        }

        {
            auto expected = expected_t{
                {0, {0, 1}},
            };

            auto set = difference(lca1, lca2);
            EXPECT_EQ(set.level(), 1);

            bool found     = false;
            std::size_t ie = 0;
            apply(set,
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                      found = true;
                  });
            EXPECT_TRUE(found);
        }

        {
            auto expected = expected_t{
                {0, {0, 2}},
                {1, {0, 2}},
            };

            auto set = self(lca1).on(0).on(1);
            EXPECT_EQ(set.level(), 1);

            bool found     = false;
            std::size_t ie = 0;
            apply(set,
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                      found = true;
                  });
            EXPECT_TRUE(found);
            EXPECT_EQ(ie, expected.size());
        }

        {
            auto expected = expected_t{
                {0, {0, 2}},
                {1, {0, 1}},
            };

            auto set = difference(self(lca1).on(0).on(1), lca2);
            EXPECT_EQ(set.level(), 1);

            bool found     = false;
            std::size_t ie = 0;
            apply(set,
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                      found = true;
                  });
            EXPECT_TRUE(found);
            EXPECT_EQ(ie, expected.size());
        }

        {
            xt::xtensor_fixed<int, xt::xshape<2>> translation{-1, -1};
            auto expected = expected_t{
                {-1, {-1, 1}},
                {0,  {-1, 0}},
            };

            // auto set = difference(translate(difference(self(lca1).on(0), lca2), translation), translate(lca1, translation));
            auto set = translate(difference(self(lca1).on(0), lca2), translation);
            EXPECT_EQ(set.level(), 1);

            bool found     = false;
            std::size_t ie = 0;
            apply(set,
                  [&](auto& i, auto& index)
                  {
                      EXPECT_EQ(expected[ie++], std::make_pair(index[0], i));
                      found = true;
                  });
            EXPECT_TRUE(found);
            EXPECT_EQ(ie, expected.size());
        }
    }

    // 1D Pathological Cases
    TEST(subset, 1d_empty_sets)
    {
        LevelCellList<1> lcl{3};
        LevelCellArray<1> empty_lca;
        LevelCellArray<1> regular_lca;

        // Empty set operations
        lcl[{}].add_interval({5, 10});
        regular_lca = lcl;

        // Union with empty set should return original
        bool found = false;
        apply(union_(empty_lca, regular_lca),
              [&](auto& i, auto)
              {
                  EXPECT_EQ(i, typename LevelCellArray<1>::interval_t(5, 10));
                  found = true;
              });
        EXPECT_TRUE(found);

        // Intersection with empty set should be empty
        found = false;
        apply(intersection(empty_lca, regular_lca),
              [&](auto&, auto)
              {
                  found = true;
              });
        EXPECT_FALSE(found);

        // Difference with empty set should return original
        found = false;
        apply(difference(regular_lca, empty_lca),
              [&](auto& i, auto)
              {
                  EXPECT_EQ(i, typename LevelCellArray<1>::interval_t(5, 10));
                  found = true;
              });
        EXPECT_TRUE(found);
    }

    TEST(subset, 1d_single_point_intervals)
    {
        LevelCellList<1> lcl{3};
        LevelCellArray<1> lca;

        // Single point intervals
        lcl[{}].add_interval({0, 1});
        lcl[{}].add_interval({2, 3});
        lcl[{}].add_interval({4, 5});
        lca = lcl;

        // Translation by 1 should create gaps
        xt::xtensor_fixed<int, xt::xshape<1>> translation{1};
        bool found = false;
        apply(difference(lca, translate(lca, translation)),
              [&](auto& i, auto)
              {
                  found = true;
                  EXPECT_TRUE(i == typename LevelCellArray<1>::interval_t(0, 1) || i == typename LevelCellArray<1>::interval_t(2, 3)
                              || i == typename LevelCellArray<1>::interval_t(4, 5));
              });
        EXPECT_TRUE(found);
    }

    TEST(subset, 1d_extreme_level_differences)
    {
        CellList<1> cl;
        CellArray<1> ca;

        // Very different levels
        cl[0][{}].add_interval({0, 1});
        cl[10][{}].add_interval({0, 1024});
        ca = {cl, true};

        // Intersection should work despite level difference
        bool found = false;
        apply(intersection(ca[0], ca[10]),
              [&](auto& i, auto)
              {
                  found = true;
                  EXPECT_EQ(i, typename CellArray<1>::interval_t(0, 1024));
              });
        EXPECT_TRUE(found);

        // Level adaptation
        found = false;
        apply(intersection(ca[0], self(ca[10]).on(0)),
              [&](auto& i, auto)
              {
                  found = true;
                  EXPECT_EQ(i, typename CellArray<1>::interval_t(0, 1));
              });
        EXPECT_TRUE(found);
    }

    TEST(subset, 1d_negative_coordinates)
    {
        LevelCellList<1> lcl{3};
        LevelCellArray<1> lca;

        lcl[{}].add_interval({-10, -5});
        lcl[{}].add_interval({-2, 3});
        lcl[{}].add_interval({8, 12});
        lca = lcl;

        xt::xtensor_fixed<int, xt::xshape<1>> translation{-15};

        // Translation to very negative values
        bool found = false;
        apply(translate(lca, translation),
              [&](auto& i, auto)
              {
                  found = true;
                  EXPECT_TRUE(i == typename LevelCellArray<1>::interval_t(-25, -20) || i == typename LevelCellArray<1>::interval_t(-17, -12)
                              || i == typename LevelCellArray<1>::interval_t(-7, -3));
              });
        EXPECT_TRUE(found);
    }

    // 2D Pathological Cases
    TEST(subset, 2d_sparse_distribution)
    {
        LevelCellList<2> lcl{3};
        LevelCellArray<2> lca;

        // Very sparse distribution
        lcl[{0}].add_interval({0, 1});
        lcl[{100}].add_interval({0, 1});
        lcl[{-50}].add_interval({200, 201});
        lca = lcl;

        xt::xtensor_fixed<int, xt::xshape<2>> translation{1, 1};

        // Self-intersection after translation should be empty for sparse data
        bool found = false;
        apply(intersection(lca, translate(lca, translation)),
              [&](auto&, auto)
              {
                  found = true;
              });
        EXPECT_FALSE(found);
    }

    TEST(subset, 2d_checkerboard_pattern)
    {
        LevelCellList<2> lcl{3};
        LevelCellArray<2> lca;

        // Checkerboard pattern
        for (int j = 0; j < 8; j += 2)
        {
            for (int i = j % 4; i < 8; i += 4)
            {
                lcl[{j}].add_interval({i, i + 1});
            }
        }
        lca = lcl;

        xt::xtensor_fixed<int, xt::xshape<2>> translation{1, 1};

        // Translation should create complementary pattern
        bool found = false;
        int count  = 0;
        apply(intersection(lca, translate(lca, translation)),
              [&](auto&, auto)
              {
                  found = true;
                  count++;
              });
        EXPECT_FALSE(found); // Should be empty due to checkerboard
    }

    TEST(subset, 2d_boundary_conditions)
    {
        CellList<2> cl;
        CellArray<2> ca;

        // Create a domain with holes
        cl[5][{16}].add_interval({0, 32});
        cl[5][{17}].add_interval({0, 8});
        cl[5][{17}].add_interval({24, 32});
        cl[5][{18}].add_interval({0, 32});
        ca = {cl, true};

        // Test boundary extraction
        xt::xtensor_fixed<int, xt::xshape<2>> directions[] = {
            {1,  0 },
            {-1, 0 },
            {0,  1 },
            {0,  -1}
        };

        for (auto& dir : directions)
        {
            auto boundary = difference(ca[5], translate(ca[5], dir));
            bool found    = false;
            apply(boundary,
                  [&](auto&, auto)
                  {
                      found = true;
                  });
            EXPECT_TRUE(found);
        }
    }

    TEST(subset, 2d_extreme_aspect_ratios)
    {
        LevelCellList<2> lcl{3};
        LevelCellArray<2> lca;

        // Very thin horizontal strips at y=0 and y=1
        lcl[{0}].add_interval({0, 1000});
        lcl[{1}].add_interval({0, 1000});

        // Very thin vertical strips at y=0 (overlapping with horizontal strips)
        for (int i = 0; i < 1000; i += 100)
        {
            lcl[{0}].add_interval({i, i + 1});
        }
        lca = lcl;

        // Intersection should find overlap regions where horizontal and vertical strips meet
        bool found = false;
        int count  = 0;
        apply(intersection(self(lca), self(lca)),
              [&](auto& i, auto& index)
              {
                  found = true;
                  count++;
                  // Should find the original intervals since self-intersection returns the original set
                  if (index[0] == 0)
                  {
                      EXPECT_TRUE(i.start >= 0 && i.end <= 1000);
                  }
              });
        EXPECT_TRUE(found);
        EXPECT_GT(count, 0);
    }

    // 3D Pathological Cases
    TEST(subset, 3d_complex_geometry)
    {
        LevelCellList<3> lcl{2};
        LevelCellArray<3> lca;

        // Create a complex 3D structure
        for (int k = 0; k < 4; ++k)
        {
            for (int j = 0; j < 4; ++j)
            {
                // Hollow cube structure
                if (k == 0 || k == 3 || j == 0 || j == 3)
                {
                    lcl[{j, k}].add_interval({0, 4});
                }
                else
                {
                    lcl[{j, k}].add_interval({0, 1});
                    lcl[{j, k}].add_interval({3, 4});
                }
            }
        }
        lca = lcl;

        // Test volume calculation through iteration
        std::size_t cell_count = 0;
        apply(self(lca),
              [&](auto& i, auto&)
              {
                  cell_count += i.size();
              });
        EXPECT_GT(cell_count, 0);

        // Test 3D translation
        xt::xtensor_fixed<int, xt::xshape<3>> translation{1, 1, 1};
        bool found = false;
        apply(intersection(lca, translate(lca, translation)),
              [&](auto&, auto)
              {
                  found = true;
              });
        EXPECT_TRUE(found);
    }

    TEST(subset, 3d_layered_structure)
    {
        LevelCellList<3> lcl{3};
        LevelCellArray<3> lca;

        // Alternating layers
        for (int k = 0; k < 8; k += 2)
        {
            for (int j = 0; j < 8; ++j)
            {
                lcl[{j, k}].add_interval({0, 8});
            }
        }
        lca = lcl;

        // Test difference between adjacent layers
        LevelCellList<3> lcl2{3};
        LevelCellArray<3> lca2;

        for (int k = 1; k < 8; k += 2)
        {
            for (int j = 0; j < 8; ++j)
            {
                lcl2[{j, k}].add_interval({0, 8});
            }
        }
        lca2 = lcl2;

        // Should be completely disjoint
        bool found = false;
        apply(intersection(lca, lca2),
              [&](auto&, auto)
              {
                  found = true;
              });
        EXPECT_FALSE(found);

        // Union should cover all layers
        found = false;
        apply(union_(lca, lca2),
              [&](auto&, auto)
              {
                  found = true;
              });
        EXPECT_TRUE(found);
    }

    TEST(subset, 3d_fractal_like_structure)
    {
        CellList<3> cl;
        CellArray<3> ca;

        // Create a Menger sponge-like structure at different levels
        cl[3][{0, 0}].add_interval({0, 9});
        cl[3][{0, 1}].add_interval({0, 3});
        cl[3][{0, 1}].add_interval({6, 9});
        cl[3][{0, 2}].add_interval({0, 9});

        cl[4][{0, 4}].add_interval({0, 18});
        cl[4][{1, 4}].add_interval({0, 6});
        cl[4][{1, 4}].add_interval({12, 18});
        cl[4][{2, 4}].add_interval({0, 18});

        ca = {cl, true};

        std::cout << self(ca[4]).on(3).to_lca() << std::endl;

        fmt::print("===============================================\n");

        // Test self-similarity at different scales
        bool found = false;
        apply(intersection(ca[3], self(ca[4]).on(3)),
              [&](auto&, auto)
              {
                  found = true;
              });
        EXPECT_TRUE(found);
    }

    TEST(subset, edge_case_translations)
    {
        LevelCellList<2> lcl{5};
        LevelCellArray<2> lca;

        // Single cell
        lcl[{16}].add_interval({16, 17});
        lca = lcl;

        // Test translation by exactly the cell size at different levels
        for (int level_offset = -2; level_offset <= 2; ++level_offset)
        {
            int target_level = 5 + level_offset;
            if (target_level >= 0)
            {
                int scale = 1 << std::abs(level_offset);
                xt::xtensor_fixed<int, xt::xshape<2>> translation;

                if (level_offset >= 0)
                {
                    translation = {scale, 0};
                }
                else
                {
                    translation = {1, 0};
                }

                bool found = false;
                apply(intersection(self(lca).on(target_level), translate(self(lca).on(target_level), translation)),
                      [&](auto&, auto)
                      {
                          found = true;
                      });
                // Should be empty for non-zero translations of single cells
                if (translation[0] != 0 || translation[1] != 0)
                {
                    EXPECT_FALSE(found);
                }
            }
        }
    }

    TEST(subset, large_coordinate_stress_test)
    {
        LevelCellList<1> lcl{10};
        LevelCellArray<1> lca;

        // Very large coordinates near integer limits
        const int large_coord = 1000000;
        lcl[{}].add_interval({large_coord, large_coord + 1024});
        lca = lcl;

        // Test operations with large coordinates
        xt::xtensor_fixed<int, xt::xshape<1>> translation{-large_coord};

        bool found = false;
        apply(translate(lca, translation),
              [&](auto& i, auto)
              {
                  EXPECT_EQ(i, typename LevelCellArray<1>::interval_t(0, 1024));
                  found = true;
              });
        EXPECT_TRUE(found);
    }

    TEST(subset, mixed_operations_stress_test)
    {
        LevelCellList<2> lcl{4};
        LevelCellArray<2> lca1, lca2;

        // Setup two overlapping patterns
        lcl[{2}].add_interval({0, 8});
        lcl[{3}].add_interval({2, 6});
        lcl[{4}].add_interval({0, 8});
        lca1 = lcl;

        lcl.clear();
        lcl[{1}].add_interval({4, 12});
        lcl[{2}].add_interval({6, 10});
        lcl[{3}].add_interval({4, 12});
        lca2 = lcl;

        // Complex nested operations
        xt::xtensor_fixed<int, xt::xshape<2>> trans1{1, 1};
        xt::xtensor_fixed<int, xt::xshape<2>> trans2{-1, 0};

        auto complex_set = union_(intersection(translate(lca1, trans1), lca2), difference(lca1, translate(lca2, trans2)));

        bool found = false;
        apply(complex_set,
              [&](auto&, auto)
              {
                  found = true;
              });
        EXPECT_TRUE(found);
    }

    TEST(subset, diff_1d_translate)
    {
        LevelCellArray<1> lca(1);
        lca.add_interval_back({0, 16}, {});
        xt::xtensor_fixed<int, xt::xshape<1>> translation{-1};

        bool never_call = true;
        apply(difference(lca, translate(lca, translation)),
              [&never_call](auto& i, auto)
              {
                  never_call = false;
                  EXPECT_EQ(i, typename LevelCellArray<1>::interval_t(15, 16));
              });
        EXPECT_FALSE(never_call);

        never_call = true;
        apply(difference(lca, translate(lca, translation).on(0)),
              [&never_call](auto&, auto)
              {
                  never_call = false;
              });
        EXPECT_TRUE(never_call);
    }

    TEST(subset, empty)
    {
        LevelCellArray<1> lca(1);
        LevelCellArray<1> lca_empty(1);
        lca.add_interval_back({0, 16}, {});
        xt::xtensor_fixed<int, xt::xshape<1>> translation{16};

        EXPECT_FALSE(lca.empty());
        EXPECT_TRUE(lca_empty.empty());

        EXPECT_TRUE(intersection(lca, lca_empty).empty());
        EXPECT_FALSE(intersection(lca, lca).empty());
        EXPECT_FALSE(difference(lca, lca_empty).empty());
        EXPECT_TRUE(intersection(lca, translate(lca, translation)).empty());
    }

}
