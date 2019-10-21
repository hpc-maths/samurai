#include <gtest/gtest.h>

#include <mure/cell_list.hpp>
#include <mure/mr/mr_config.hpp>

namespace mure
{
    TEST(cell_list, constructor)
    {
        constexpr size_t dim = 2;
        using Config = MRConfig<dim>;

        CellList<Config> cell_list;

        std::cout << cell_list << "\n";
    }
}