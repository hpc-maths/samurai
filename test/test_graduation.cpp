#include <algorithm>

#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/graduation.hpp>

namespace samurai
{
    TEST(graduation, dim_1)
    {
        constexpr size_t dim = 1;
        CellList<dim> cl;
        cl[0][{}].add_interval({1, 2});
        cl[5][{}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }

    TEST(graduation, dim_2)
    {
        constexpr size_t dim = 2;
        CellList<dim> cl;
        cl[0][{1}].add_interval({1, 2});
        cl[5][{0}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }

    TEST(graduation, dim_3)
    {
        constexpr size_t dim = 3;
        CellList<dim> cl;
        cl[0][{1, 1}].add_interval({1, 2});
        cl[5][{0, 0}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }
}
