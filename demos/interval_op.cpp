#include <mure/intervals_op.hpp>
#include <mure/level_cell_array.hpp>
#include <mure/level_cell_list.hpp>
#include <mure/mr_config.hpp>

int main()
{
    constexpr std::size_t dim = 1;
    using config = mure::MRConfig<dim>;

    mure::LevelCellList<config> level_cell_list_1;
    level_cell_list_1[{}].add_interval({0, 2});
    level_cell_list_1[{}].add_interval({4, 7});
    mure::LevelCellArray<config> level_cell_array_1{level_cell_list_1};

    mure::LevelCellList<config> level_cell_list_2;
    level_cell_list_2[{}].add_interval({1, 5});
    mure::LevelCellArray<config> level_cell_array_2{level_cell_list_2};

    std::cout << "intersection\n";
    mure::intersection(level_cell_array_1, level_cell_array_2);
    std::cout << "union\n";
    mure::union_(level_cell_array_1, level_cell_array_2);
    std::cout << "difference\n";
    mure::difference(level_cell_array_1, level_cell_array_2);
}