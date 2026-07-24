// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <string>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include <gtest/gtest.h>

#include <samurai/cell_list.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/subset/node.hpp>

#include <samurai/subset/dynamic/dynamic.hpp>

namespace samurai
{
    namespace
    {
        // The point set produced by traversing a set, as (index, interval)
        // pairs. Comparing these for a static and a dynamic expression is the
        // equivalence check throughout this file.
        template <class Interval>
        using traversal_t = std::vector<std::pair<std::string, Interval>>;

        template <class Func>
        auto record(Func&& traverse)
        {
            using interval_t = typename LevelCellArray<1>::interval_t;
            traversal_t<interval_t> out;
            traverse(
                [&](const auto& i, const auto& index)
                {
                    std::string idx;
                    for (std::size_t d = 0; d < index.size(); ++d)
                    {
                        idx += std::to_string(index[d]) + ",";
                    }
                    out.emplace_back(idx, i);
                });
            return out;
        }

        template <class Set>
        auto traverse_static(const Set& set)
        {
            return record(
                [&](auto&& func)
                {
                    apply(set, func);
                });
        }

        template <std::size_t dim, class TInterval>
        auto traverse_dynamic(const DynamicSet<dim, TInterval>& set)
        {
            return record(
                [&](auto&& func)
                {
                    set(func);
                });
        }

        LevelCellArray<1> lca1(std::size_t level, std::initializer_list<std::pair<int, int>> intervals)
        {
            LevelCellList<1> lcl{level};
            for (auto [a, b] : intervals)
            {
                lcl[{}].add_interval({a, b});
            }
            return lcl;
        }

        LevelCellArray<2> lca2(std::size_t level, std::initializer_list<std::tuple<int, int, int>> boxes)
        {
            LevelCellList<2> lcl{level};
            for (auto [y, x0, x1] : boxes)
            {
                lcl[{y}].add_interval({x0, x1});
            }
            return LevelCellArray<2>(lcl);
        }
    } // namespace

    ////////////////////////////////////////////////////////////////////////
    //// 1D equivalence with the static algebra
    ////////////////////////////////////////////////////////////////////////

    TEST(subset_dynamic, leaf)
    {
        auto a = lca1(0, {{0, 2}, {5, 9}, {12, 14}});
        EXPECT_EQ(traverse_static(self(a)), traverse_dynamic(dyn::self(a)));
    }

    TEST(subset_dynamic, intersection_1d)
    {
        auto a = lca1(2, {{0, 5}, {8, 12}});
        auto b = lca1(2, {{3, 9}, {11, 20}});
        auto c = lca1(2, {{4, 7}, {10, 30}});

        EXPECT_EQ(traverse_static(intersection(self(a), self(b), self(c))),
                  traverse_dynamic(dyn::intersection(dyn::self(a), dyn::self(b), dyn::self(c))));
    }

    TEST(subset_dynamic, union_1d)
    {
        auto a = lca1(2, {{0, 5}, {8, 12}});
        auto b = lca1(2, {{3, 9}, {11, 20}});
        auto c = lca1(2, {{4, 7}, {10, 30}});

        EXPECT_EQ(traverse_static(union_(self(a), self(b), self(c))),
                  traverse_dynamic(dyn::union_(dyn::self(a), dyn::self(b), dyn::self(c))));
    }

    TEST(subset_dynamic, difference_1d)
    {
        auto a = lca1(2, {{0, 5}, {8, 12}});
        auto b = lca1(2, {{3, 9}, {11, 20}});
        auto c = lca1(2, {{4, 7}, {10, 30}});

        EXPECT_EQ(traverse_static(difference(self(a), self(b), self(c))),
                  traverse_dynamic(dyn::difference(dyn::self(a), dyn::self(b), dyn::self(c))));
    }

    TEST(subset_dynamic, nested_1d)
    {
        auto a = lca1(2, {{0, 5}, {8, 12}});
        auto b = lca1(2, {{3, 9}, {11, 20}});
        auto c = lca1(2, {{4, 7}, {10, 30}});

        auto stat = union_(intersection(self(a), self(b)).on(1), self(c));
        auto dynv = dyn::union_(dyn::intersection(dyn::self(a), dyn::self(b)).on(1), dyn::self(c));
        EXPECT_EQ(traverse_static(stat), traverse_dynamic(dynv));
    }

    TEST(subset_dynamic, mixed_levels_1d)
    {
        auto a = lca1(2, {{0, 5}, {8, 12}});
        auto d = lca1(4, {{0, 40}});
        EXPECT_EQ(traverse_static(intersection(self(a), self(d))),
                  traverse_dynamic(dyn::intersection(dyn::self(a), dyn::self(d))));
    }

    ////////////////////////////////////////////////////////////////////////
    //// 2D equivalence (exercises the runtime -> compile-time dimension switch)
    ////////////////////////////////////////////////////////////////////////

    TEST(subset_dynamic, intersection_2d)
    {
        auto a = lca2(3, {{0, 0, 5}, {1, 2, 8}, {2, 0, 4}});
        auto b = lca2(3, {{0, 3, 9}, {1, 0, 6}, {2, 1, 3}});
        EXPECT_EQ(traverse_static(intersection(self(a), self(b))),
                  traverse_dynamic(dyn::intersection(dyn::self(a), dyn::self(b))));
    }

    TEST(subset_dynamic, union_2d)
    {
        auto a = lca2(3, {{0, 0, 5}, {1, 2, 8}, {2, 0, 4}});
        auto b = lca2(3, {{0, 3, 9}, {1, 0, 6}, {2, 1, 3}});
        auto c = lca2(3, {{1, 1, 7}, {2, 0, 10}});
        EXPECT_EQ(traverse_static(union_(self(a), self(b), self(c))),
                  traverse_dynamic(dyn::union_(dyn::self(a), dyn::self(b), dyn::self(c))));
    }

    TEST(subset_dynamic, difference_2d)
    {
        auto a = lca2(3, {{0, 0, 5}, {1, 2, 8}, {2, 0, 4}});
        auto b = lca2(3, {{0, 3, 9}, {1, 0, 6}, {2, 1, 3}});
        EXPECT_EQ(traverse_static(difference(self(a), self(b))),
                  traverse_dynamic(dyn::difference(dyn::self(a), dyn::self(b))));
    }

    TEST(subset_dynamic, projection_2d)
    {
        auto a = lca2(3, {{0, 0, 5}, {1, 2, 8}, {2, 0, 4}});
        auto b = lca2(3, {{0, 3, 9}, {1, 0, 6}, {2, 1, 3}});
        EXPECT_EQ(traverse_static(intersection(self(a), self(b)).on(2)),
                  traverse_dynamic(dyn::intersection(dyn::self(a), dyn::self(b)).on(2)));
    }

    TEST(subset_dynamic, translate_2d)
    {
        auto a = lca2(3, {{0, 0, 5}, {1, 2, 8}, {2, 0, 4}});
        auto b = lca2(3, {{0, 3, 9}, {1, 0, 6}, {2, 1, 3}});

        xt::xtensor_fixed<int, xt::xshape<2>> t{1, 2};
        EXPECT_EQ(traverse_static(intersection(translate(self(a), t), self(b))),
                  traverse_dynamic(dyn::intersection(dyn::translate(dyn::self(a), t), dyn::self(b))));
    }

    TEST(subset_dynamic, expand_2d)
    {
        auto a = lca2(3, {{0, 0, 5}, {1, 2, 8}, {2, 0, 4}});
        auto c = lca2(3, {{1, 1, 7}, {2, 0, 10}});
        EXPECT_EQ(traverse_static(intersection(nestedExpand(self(a), 1), self(c))),
                  traverse_dynamic(dyn::intersection(dyn::expand(dyn::self(a), 1), dyn::self(c))));
    }

    ////////////////////////////////////////////////////////////////////////
    //// Dynamic-specific behaviour
    ////////////////////////////////////////////////////////////////////////

    // The number of operands is a runtime vector: the natural entry point for
    // bindings. Must match the variadic form.
    TEST(subset_dynamic, runtime_vector_api)
    {
        auto a = lca1(0, {{0, 10}});
        auto b = lca1(0, {{5, 15}});
        auto c = lca1(0, {{2, 8}});

        using set_t = DynamicSet<1, LevelCellArray<1>::interval_t>;
        std::vector<set_t> operands{dyn::self(a), dyn::self(b), dyn::self(c)};

        EXPECT_EQ(traverse_dynamic(dyn::intersection(operands)),
                  traverse_dynamic(dyn::intersection(dyn::self(a), dyn::self(b), dyn::self(c))));
    }

    // clone() must yield an independent tree giving the same result.
    TEST(subset_dynamic, clone_is_independent)
    {
        auto a = lca1(0, {{0, 10}});
        auto b = lca1(0, {{5, 15}});

        auto original = dyn::intersection(dyn::self(a), dyn::self(b));
        auto copy     = original.clone();

        auto expected = traverse_dynamic(original);
        EXPECT_EQ(expected, traverse_dynamic(copy));
        EXPECT_EQ(expected, traverse_dynamic(original)); // original still traversable
    }

    // to_lca() of a dynamic expression equals the static one.
    TEST(subset_dynamic, to_lca)
    {
        auto a = lca1(2, {{0, 5}, {8, 12}});
        auto b = lca1(2, {{3, 9}, {11, 20}});

        auto static_lca  = intersection(self(a), self(b)).on(1).to_lca();
        auto dynamic_lca = dyn::intersection(dyn::self(a), dyn::self(b)).on(1).to_lca();
        EXPECT_EQ(static_lca, dynamic_lca);
    }

} // namespace samurai
