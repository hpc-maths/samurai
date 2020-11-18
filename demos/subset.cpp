// #include <samurai/intervals_operator.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/mr/mr_config.hpp>
// #include <samurai/subset.hpp>
#include <samurai/subset/subset_op.hpp>

int main()
{
    constexpr size_t dim = 2;

    samurai::LevelCellList<dim> lcl1(1);

    lcl1[{-4}].add_interval({-4, 4});
    lcl1[{-3}].add_interval({-4, 4});
    lcl1[{-2}].add_interval({-4, -1});
    lcl1[{-2}].add_interval({1, 4});
    lcl1[{-1}].add_interval({-4, -2});
    lcl1[{-1}].add_interval({2, 4});
    lcl1[{0}].add_interval({-4, -2});
    lcl1[{0}].add_interval({2, 4});
    lcl1[{1}].add_interval({-4, -1});
    lcl1[{1}].add_interval({1, 4});
    lcl1[{2}].add_interval({-4, 4});
    lcl1[{3}].add_interval({-4, 4});

    samurai::LevelCellList<dim> lcl2(1);
    lcl2[{-2}].add_interval({-1, 1});
    lcl2[{-1}].add_interval({-2, 2});
    lcl2[{0}].add_interval({-2, 2});
    lcl2[{1}].add_interval({-1, 1});

    samurai::LevelCellArray<dim> lca1{lcl1};
    samurai::LevelCellArray<dim> lca2{lcl2};

    std::cout << lca1 << "\n";
    std::cout << lca2 << "\n";
    auto expr = samurai::union_(lca1, lca2).on(0);
    expr([](auto &index, auto &interval, auto &) {
        std::cout << index[0] << " " << interval[0] << "\n";
    });

    return 0;
}