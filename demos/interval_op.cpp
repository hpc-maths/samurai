#include <mure/subset.hpp>
#include <mure/intervals_operator.hpp>
#include <mure/level_cell_array.hpp>
#include <mure/level_cell_list.hpp>
#include <mure/mr_config.hpp>

int main()
{
    constexpr std::size_t dim = 1;
    using config = mure::MRConfig<dim>;

    mure::LevelCellList<config> level_cell_list_1;
    level_cell_list_1[{}].add_interval({2, 3});
    level_cell_list_1[{}].add_interval({5, 6});
    // level_cell_list_1[{}].add_interval({4, 7});
    mure::LevelCellArray<config> level_cell_array_1{level_cell_list_1};

    mure::LevelCellList<config> level_cell_list_2;
    level_cell_list_2[{}].add_interval({5, 8});
    // level_cell_list_2[{}].add_interval({1, 5});
    mure::LevelCellArray<config> level_cell_array_2{level_cell_list_2};

    mure::LevelCellList<config> level_cell_list_3;
    level_cell_list_3[{}].add_interval({3, 10});
    mure::LevelCellArray<config> level_cell_array_3{level_cell_list_3};

    // auto expr = mure::union_(mure::_1, mure::_2, mure::_3);
    auto expr = mure::intersection(mure::_1, mure::_2);
    // auto set = mure::make_subset<config>(expr, level_cell_array_1, level_cell_array_2);
    auto set = mure::make_subset<config>(expr, 0, {0, 0}, level_cell_array_1, level_cell_array_2);

    set.apply([&](auto& index_yz, auto& interval, auto& interval_index)
                {
                    std::cout << index_yz << " " << interval << "\n";
                    std::cout << interval_index << "\n";
                    level_cell_array_1[interval_index[0, 0]].index = 42;
                    // std::cout << intervals[0].get() << "\n";
                    // std::cout << intervals[1].get() << "\n";
                });
    //set.apply([](auto& index_yz, auto& interval){std::cout << index_yz << " " << interval << "\n";});
    std::cout << level_cell_array_1[1] << "\n";

    /////////////////////////////////////////
    //
    // 2D
    //
    /////////////////////////////////////////
    // constexpr std::size_t dim = 2;
    // using config = mure::MRConfig<dim>;

    // mure::LevelCellList<config> level_cell_list_1;
    // level_cell_list_1.extend({0}, {4});
    // level_cell_list_1[0].add_interval({0, 2});
    // level_cell_list_1[0].add_interval({4, 7});
    // mure::LevelCellArray<config> level_cell_array_1{level_cell_list_1};

    // mure::LevelCellList<config> level_cell_list_2;
    // level_cell_list_2.extend({0}, {4});
    // level_cell_list_2[0].add_interval({0, 2});
    // level_cell_list_2[1].add_interval({0, 2});

    // mure::LevelCellArray<config> level_cell_array_2{level_cell_list_2};


    // auto expr = mure::difference(mure::_1, mure::_2);
    // auto set = mure::make_subset<config>(expr, level_cell_array_1, level_cell_array_2);
    // set.apply([](auto& index_yz, auto& interval){std::cout << index_yz << " " << interval << "\n";});

    /////////////////////////////////////////
    //
    // 3D
    //
    /////////////////////////////////////////
    // constexpr std::size_t dim = 3;
    // using config = mure::MRConfig<dim>;

    // mure::LevelCellList<config> level_cell_list_1;
    // level_cell_list_1.extend({0, 0}, {4, 4});
    // level_cell_list_1[{0, 0}].add_interval({0, 2});
    // // level_cell_list_1[0].add_interval({4, 7});
    // mure::LevelCellArray<config> level_cell_array_1{level_cell_list_1};

    // mure::LevelCellList<config> level_cell_list_2;
    // level_cell_list_2.extend({0, 0}, {4, 4});
    // level_cell_list_2[{0, 0}].add_interval({0, 5});
    // level_cell_list_2[{0, 1}].add_interval({0, 2});

    // mure::LevelCellArray<config> level_cell_array_2{level_cell_list_2};

    // auto expr = mure::intersection(mure::_2, mure::_1);
    // auto level_cell_arrays = std::tie(level_cell_array_1,
    //                                   level_cell_array_2);

    // auto set = mure::make_subset<config>(expr, level_cell_array_1, level_cell_array_2);
    // set.apply([](auto& index_yz, auto& interval){std::cout << index_yz << " " << interval << "\n";});
}