#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>

#include <samurai/level_cell_list.hpp>

namespace samurai
{
    TEST(level_cell_list, add_interval)
    {
        constexpr size_t dim = 2;
        LevelCellList<dim> lcl;
        lcl[{0}].add_interval({-3, 3});
    }
}
