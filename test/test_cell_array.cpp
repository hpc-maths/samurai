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
}
