// #include <mure/intervals_operator.hpp>
#include <mure/level_cell_array.hpp>
#include <mure/level_cell_list.hpp>
#include <mure/mr/mr_config.hpp>
// #include <mure/subset.hpp>
#include <mure/subset/subset_op.hpp>

int main()
{
    constexpr size_t dim = 2;

    mure::LevelCellList<dim> lcl1(1);

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

    mure::LevelCellList<dim> lcl2(1);
    lcl2[{-2}].add_interval({-1, 1});
    lcl2[{-1}].add_interval({-2, 2});
    lcl2[{0}].add_interval({-2, 2});
    lcl2[{1}].add_interval({-1, 1});

    mure::LevelCellArray<dim> lca1{lcl1};
    mure::LevelCellArray<dim> lca2{lcl2};

    std::cout << lca1 << "\n";
    std::cout << lca2 << "\n";
    auto expr = mure::union_(lca1, lca2).on(0);
    expr([](auto &index, auto &interval, auto &) {
        std::cout << index[0] << " " << interval[0] << "\n";
    });

    return 0;
}