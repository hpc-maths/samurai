#include <gtest/gtest.h>

#include <samurai/cell_list.hpp>

namespace samurai
{
    TEST(cell_list, constructor)
    {
        constexpr size_t dim = 2;

        CellList<dim> cell_list;
    }
}
