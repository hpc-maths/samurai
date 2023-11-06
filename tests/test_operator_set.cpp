#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <samurai/interval.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/subset/subset_op.hpp>

#include "test_common.hpp"

namespace samurai
{
    // RC_GTEST_PROP(operator, creation,(std::vector<Interval<int>> i))
    // {
    //     LevelCellList<1> lcl;
    //     for (auto &ii : i)
    //         lcl[{}].add_interval(std::move(ii));

    //     LevelCellArray<1> lca{lcl};

    //     std::cout << lca << "\n\n";
    // }

    template <std::size_t size>
    auto give_me_lca_1d()
    {
        constexpr std::size_t dim = 1;
        const auto level          = *rc::gen::container<std::array<std::size_t, size>>(rc::gen::inRange(2, 12));

        std::vector<LevelCellList<dim>> lcl;
        std::vector<LevelCellArray<dim>> lca;

        for (std::size_t i = 0; i < level.size(); ++i)
        {
            lcl.push_back({i});
            std::size_t level_size = *rc::gen::inRange<std::size_t>(0, 100);
            const auto ints        = *rc::gen::container<std::vector<int>>(level_size, rc::gen::inRange(-20 * (1 << i), 20 * (1 << i)));

            for (auto& ii : ints)
            {
                lcl[i][{}].add_point(ii);
            }

            lca.push_back({lcl[i]});
        }
        return lca;
    }

    template <std::size_t size>
    auto give_me_lca_2d()
    {
        constexpr std::size_t dim = 2;
        const auto level          = *rc::gen::container<std::array<std::size_t, size>>(rc::gen::inRange(2, 12));

        std::vector<LevelCellList<dim>> lcl;
        std::vector<LevelCellArray<dim>> lca;

        for (std::size_t i = 0; i < level.size(); ++i)
        {
            lcl.push_back({i});
            std::size_t level_size = *rc::gen::inRange<std::size_t>(0, 100);
            const auto ints_x      = *rc::gen::container<std::vector<int>>(level_size, rc::gen::inRange(-20 * (1 << i), 20 * (1 << i)));
            const auto ints_y      = *rc::gen::container<std::vector<int>>(level_size, rc::gen::inRange(-20 * (1 << i), 20 * (1 << i)));

            for (std::size_t ii = 0; ii < level_size; ++ii)
            {
                lcl[i][{ints_y[ii]}].add_point(ints_x[ii]);
            }

            lca.push_back({lcl[i]});
        }
        return lca;
    }

    RC_GTEST_PROP(operator, distributive_law_intersection_1d, ())
    {
        auto lca = give_me_lca_1d<3>();

        const std::size_t ref_level = *rc::gen::inRange<std::size_t>(2, 12);

        auto expr1 = intersection(lca[0], union_(lca[1], lca[2])).on(ref_level);
        auto expr2 = union_(intersection(lca[0], lca[1]), intersection(lca[0], lca[2])).on(ref_level);

        LevelCellList<1> lcl1{ref_level};
        expr1(
            [&](const auto& interval, auto&)
            {
                lcl1[{}].add_interval(interval);
            });

        LevelCellList<1> lcl2{ref_level};
        expr2(
            [&](const auto& interval, auto&)
            {
                lcl2[{}].add_interval(interval);
            });

        LevelCellArray<1> lca1{lcl1}, lca2{lcl2};
        RC_ASSERT(lca1 == lca2);
    }

    RC_GTEST_PROP(operator, distributive_law_intersection_2d, ())
    {
        auto lca = give_me_lca_2d<3>();

        const std::size_t ref_level = *rc::gen::inRange<std::size_t>(2, 12);

        auto expr1 = intersection(lca[0], union_(lca[1], lca[2])).on(ref_level);
        auto expr2 = union_(intersection(lca[0], lca[1]), intersection(lca[0], lca[2])).on(ref_level);

        LevelCellList<2> lcl1{ref_level};
        expr1(
            [&](const auto& interval, const auto& index)
            {
                lcl1[index].add_interval(interval);
            });

        LevelCellList<2> lcl2{ref_level};
        expr2(
            [&](const auto& interval, const auto& index)
            {
                lcl2[index].add_interval(interval);
            });

        LevelCellArray<2> lca1{lcl1}, lca2{lcl2};
        RC_ASSERT(lca1 == lca2);
    }

    RC_GTEST_PROP(operator, distributive_law_union_1d, ())
    {
        auto lca = give_me_lca_1d<3>();

        const std::size_t ref_level = *rc::gen::inRange<std::size_t>(2, 12);

        auto expr1 = union_(lca[0], intersection(lca[1], lca[2])).on(ref_level);
        auto expr2 = intersection(union_(lca[0], lca[1]), union_(lca[0], lca[2])).on(ref_level);

        LevelCellList<1> lcl1{ref_level};
        expr1(
            [&](const auto& interval, auto&)
            {
                lcl1[{}].add_interval(interval);
            });

        LevelCellList<1> lcl2{ref_level};
        expr2(
            [&](const auto& interval, auto&)
            {
                lcl2[{}].add_interval(interval);
            });

        LevelCellArray<1> lca1{lcl1}, lca2{lcl2};
        RC_ASSERT(lca1 == lca2);
    }

    RC_GTEST_PROP(operator, distributive_law_union_2d, ())
    {
        auto lca = give_me_lca_2d<3>();

        const std::size_t ref_level = *rc::gen::inRange<std::size_t>(2, 12);

        auto expr1 = union_(lca[0], intersection(lca[1], lca[2])).on(ref_level);
        auto expr2 = intersection(union_(lca[0], lca[1]), union_(lca[0], lca[2])).on(ref_level);

        LevelCellList<2> lcl1{ref_level};
        expr1(
            [&](const auto& interval, const auto& index)
            {
                lcl1[index].add_interval(interval);
            });

        LevelCellList<2> lcl2{ref_level};
        expr2(
            [&](const auto& interval, const auto& index)
            {
                lcl2[index].add_interval(interval);
            });

        LevelCellArray<2> lca1{lcl1}, lca2{lcl2};
        RC_ASSERT(lca1 == lca2);
    }

    RC_GTEST_PROP(operator, de_morgan_law_intersection_1d, ())
    {
        using box_t = Box<int, 1>;
        auto lca    = give_me_lca_1d<3>();
        LevelCellArray<1> lca_c{
            0,
            box_t{-20, 20}
        };

        const std::size_t ref_level = *rc::gen::inRange<std::size_t>(2, 12);

        auto expr1 = difference(lca_c, intersection(lca[0], intersection(lca[1], lca[2]))).on(ref_level);
        auto expr2 = union_(difference(lca_c, lca[0]), union_(difference(lca_c, lca[1]), difference(lca_c, lca[2]))).on(ref_level);

        LevelCellList<1> lcl1{ref_level};
        expr1(
            [&](const auto& interval, auto&)
            {
                lcl1[{}].add_interval(interval);
            });

        LevelCellList<1> lcl2{ref_level};
        expr2(
            [&](const auto& interval, auto&)
            {
                lcl2[{}].add_interval(interval);
            });

        LevelCellArray<1> lca1{lcl1}, lca2{lcl2};
        RC_ASSERT(lca1 == lca2);
    }

    RC_GTEST_PROP(operator, de_morgan_law_intersection_2d, ())
    {
        using box_t = Box<int, 2>;
        auto lca    = give_me_lca_2d<3>();
        LevelCellArray<2> lca_c{
            0,
            box_t{{-20, -20}, {20, 20}}
        };

        const std::size_t ref_level = *rc::gen::inRange<std::size_t>(2, 12);

        auto expr1 = difference(lca_c, intersection(lca[0], intersection(lca[1], lca[2]))).on(ref_level);
        auto expr2 = union_(difference(lca_c, lca[0]), union_(difference(lca_c, lca[1]), difference(lca_c, lca[2]))).on(ref_level);

        LevelCellList<2> lcl1{ref_level};
        expr1(
            [&](const auto& interval, const auto& index)
            {
                lcl1[index].add_interval(interval);
            });

        LevelCellList<2> lcl2{ref_level};
        expr2(
            [&](const auto& interval, const auto& index)
            {
                lcl2[index].add_interval(interval);
            });

        LevelCellArray<2> lca1{lcl1}, lca2{lcl2};
        RC_ASSERT(lca1 == lca2);
    }

    RC_GTEST_PROP(operator, de_morgan_law_union_1d, ())
    {
        using box_t = Box<int, 1>;
        auto lca    = give_me_lca_1d<3>();
        LevelCellArray<1> lca_c{
            0,
            box_t{-20, 20}
        };

        const std::size_t ref_level = *rc::gen::inRange<std::size_t>(2, 12);

        auto expr1 = difference(lca_c, union_(lca[0], union_(lca[1], lca[2]))).on(ref_level);
        auto expr2 = intersection(difference(lca_c, lca[0]), intersection(difference(lca_c, lca[1]), difference(lca_c, lca[2]))).on(ref_level);

        LevelCellList<1> lcl1{ref_level};
        expr1(
            [&](const auto& interval, auto&)
            {
                lcl1[{}].add_interval(interval);
            });

        LevelCellList<1> lcl2{ref_level};
        expr2(
            [&](const auto& interval, auto&)
            {
                lcl2[{}].add_interval(interval);
            });

        LevelCellArray<1> lca1{lcl1}, lca2{lcl2};
        RC_ASSERT(lca1 == lca2);
    }

    RC_GTEST_PROP(operator, de_morgan_law_union_2d, ())
    {
        using box_t = Box<int, 2>;
        auto lca    = give_me_lca_2d<3>();
        LevelCellArray<2> lca_c{
            0,
            box_t{{-20, -20}, {20, 20}}
        };

        const std::size_t ref_level = *rc::gen::inRange<std::size_t>(2, 12);

        auto expr1 = difference(lca_c, union_(lca[0], union_(lca[1], lca[2]))).on(ref_level);
        auto expr2 = intersection(difference(lca_c, lca[0]), intersection(difference(lca_c, lca[1]), difference(lca_c, lca[2]))).on(ref_level);

        LevelCellList<2> lcl1{ref_level};
        expr1(
            [&](const auto& interval, const auto& index)
            {
                lcl1[index].add_interval(interval);
            });

        LevelCellList<2> lcl2{ref_level};
        expr2(
            [&](const auto& interval, const auto& index)
            {
                lcl2[index].add_interval(interval);
            });

        LevelCellArray<2> lca1{lcl1}, lca2{lcl2};
        RC_ASSERT(lca1 == lca2);
    }
}
