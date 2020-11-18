#include <gtest/gtest.h>

#include <samurai/cell_list.hpp>
#include <samurai/mr/mr_config.hpp>

namespace samurai
{
    TEST(cell_list, constructor)
    {
        constexpr size_t dim = 2;
        using Config = MRConfig<dim>;

        CellList<Config> cell_list;

        std::cout << cell_list << "\n";
    }
}