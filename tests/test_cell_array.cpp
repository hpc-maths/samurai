#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

namespace samurai
{
    TEST(cell_array, iterator)
    {
        constexpr size_t dim = 2;

        CellList<dim> cell_list;
        cell_list[1][{1}].add_interval({2, 5});
        cell_list[2][{5}].add_interval({-2, 8});
        cell_list[2][{5}].add_interval({9, 10});
        cell_list[2][{6}].add_interval({10, 12});

        CellArray<dim> cell_array(cell_list);
        using interval_t = typename CellArray<dim>::interval_t;

        auto it = cell_array.begin();

        EXPECT_EQ(*it, (interval_t{2, 5, -2}));

        it += 2;

        EXPECT_EQ(*it, (interval_t{9, 10, 4}));

        it += 5;
        EXPECT_EQ(it, cell_array.end());

        auto itr = cell_array.rbegin();

        EXPECT_EQ(*itr, (interval_t{10, 12, 4}));

        itr += 2;

        EXPECT_EQ(*itr, (interval_t{-2, 8, 5}));

        itr += 5;
        EXPECT_EQ(itr, cell_array.rend());
    }

    TEST(cell_array, get_interval)
    {
        constexpr size_t dim = 2;

        CellList<dim> cell_list;

        cell_list[1][{1}].add_interval({2, 5});
        cell_list[2][{5}].add_interval({-2, 8});
        cell_list[2][{5}].add_interval({9, 10});
        cell_list[2][{6}].add_interval({10, 12});

        CellArray<dim> cell_array(cell_list);
        using interval_t = typename CellArray<dim>::interval_t;

        EXPECT_EQ(cell_array.get_interval(2, {0, 3}, 5), (interval_t{-2, 8, 5}));

        xt::xtensor_fixed<int, xt::xshape<1>> index{10};
        EXPECT_EQ(cell_array.get_interval(2, {0, 3}, index / 2), (interval_t{-2, 8, 5}));

        // TODO : nothing is done for get_interval has no answer
        // interval_t unvalid{0, 0, 0};
        // unvalid.step = 0;
        // EXPECT_EQ(cell_array.get_interval(2, {0, 3}, index / 2 + 1), unvalid);

        EXPECT_EQ(cell_array.get_interval(2, {10, 11}, index / 2 + 1), (interval_t{10, 12, 4}));

        xt::xtensor_fixed<int, xt::xshape<2>> coords{1, 2};
        EXPECT_EQ(cell_array.get_interval(2, 2 * coords + 1), (interval_t{-2, 8, 5}));
    }

    TEST(cell_array, get_index)
    {
        constexpr size_t dim = 2;

        CellList<dim> cell_list;

        cell_list[1][{1}].add_interval({2, 5});
        cell_list[2][{5}].add_interval({-2, 8});
        cell_list[2][{5}].add_interval({9, 10});
        cell_list[2][{6}].add_interval({10, 12});

        CellArray<dim> cell_array(cell_list);
        EXPECT_EQ(cell_array.get_index(2, 0, 5), 5);

        xt::xtensor_fixed<int, xt::xshape<1>> index{10};
        EXPECT_EQ(cell_array.get_index(2, 3, index / 2), 8);

        // TODO : nothing is done for get_index has no answer
        // EXPECT_EQ(cell_array.get_index(2, 0, index / 2 + 1), 0);

        EXPECT_EQ(cell_array.get_index(2, 10, index / 2 + 1), 14);

        xt::xtensor_fixed<int, xt::xshape<2>> coords{1, 2};
        EXPECT_EQ(cell_array.get_index(2, 2 * coords + 1), 8);
    }

    TEST(cell_array, get_cell)
    {
        constexpr size_t dim = 2;

        CellList<dim> cell_list;

        cell_list[1][{1}].add_interval({2, 5});
        cell_list[2][{5}].add_interval({-2, 8});
        cell_list[2][{5}].add_interval({9, 10});
        cell_list[2][{6}].add_interval({10, 12});

        CellArray<dim> cell_array(cell_list);
        using cell_t   = typename CellArray<dim>::cell_t;
        using coords_t = typename cell_t::coords_t;

        coords_t origin_point{0, 0};
        double scaling_factor = 1;

        EXPECT_EQ(cell_array.get_cell(2, 0, 5), (cell_t(origin_point, scaling_factor, 2, 0, 5, 5)));

        xt::xtensor_fixed<int, xt::xshape<1>> index{10};
        EXPECT_EQ(cell_array.get_cell(2, 3, index / 2), (cell_t(origin_point, scaling_factor, 2, 3, 5, 8)));

        // TODO : nothing is done for get_cell has no answer
        // EXPECT_EQ(cell_array.get_cell(2, 0, index / 2 + 1), (cell_t(2, 0, 6, 0)));

        EXPECT_EQ(cell_array.get_cell(2, 10, index / 2 + 1), (cell_t(origin_point, scaling_factor, 2, 10, 6, 14)));

        xt::xtensor_fixed<int, xt::xshape<2>> coords{1, 2};
        EXPECT_EQ(cell_array.get_cell(2, 2 * coords + 1), (cell_t(origin_point, scaling_factor, 2, 3, 5, 8)));
    }
}
