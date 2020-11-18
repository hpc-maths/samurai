#include <samurai/subset.hpp>
#include <samurai/intervals_operator.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/mr_config.hpp>

int main()
{
    constexpr std::size_t dim = 1;
    using config = samurai::MRConfig<dim>;

    samurai::LevelCellList<config> level_cell_list_1;
    level_cell_list_1[{}].add_interval({2, 3});
    level_cell_list_1[{}].add_interval({5, 6});
    // level_cell_list_1[{}].add_interval({4, 7});
    samurai::LevelCellArray<config> level_cell_array_1{level_cell_list_1};

    samurai::LevelCellList<config> level_cell_list_2;
    level_cell_list_2[{}].add_interval({5, 8});
    // level_cell_list_2[{}].add_interval({1, 5});
    samurai::LevelCellArray<config> level_cell_array_2{level_cell_list_2};

    samurai::LevelCellList<config> level_cell_list_3;
    level_cell_list_3[{}].add_interval({3, 10});
    samurai::LevelCellArray<config> level_cell_array_3{level_cell_list_3};

    // auto expr = samurai::union_(samurai::_1, samurai::_2, samurai::_3);
    auto expr = samurai::intersection(samurai::_1, samurai::_2);
    // auto set = samurai::make_subset<config>(expr, level_cell_array_1, level_cell_array_2);
    auto set = samurai::make_subset<config>(expr, 0, {0, 0}, level_cell_array_1, level_cell_array_2);

    set.apply([&](auto& index_yz, auto& interval, auto& interval_index)
                {
                    std::cout << index_yz << " " << interval << "\n";
                    std::cout << interval_index << "\n";
                    level_cell_array_1[0][interval_index[0, 0]].index = 42;
                    // std::cout << intervals[0].get() << "\n";
                    // std::cout << intervals[1].get() << "\n";
                });
    //set.apply([](auto& index_yz, auto& interval){std::cout << index_yz << " " << interval << "\n";});
    std::cout << level_cell_array_1[0][1] << "\n";

    /////////////////////////////////////////
    //
    // 2D
    //
    /////////////////////////////////////////
    // constexpr std::size_t dim = 2;
    // using config = samurai::MRConfig<dim>;

    // samurai::LevelCellList<config> level_cell_list_1;
    // level_cell_list_1.extend({0}, {4});
    // level_cell_list_1[0].add_interval({0, 2});
    // level_cell_list_1[0].add_interval({4, 7});
    // samurai::LevelCellArray<config> level_cell_array_1{level_cell_list_1};

    // samurai::LevelCellList<config> level_cell_list_2;
    // level_cell_list_2.extend({0}, {4});
    // level_cell_list_2[0].add_interval({0, 2});
    // level_cell_list_2[1].add_interval({0, 2});

    // samurai::LevelCellArray<config> level_cell_array_2{level_cell_list_2};


    // auto expr = samurai::difference(samurai::_1, samurai::_2);
    // auto set = samurai::make_subset<config>(expr, level_cell_array_1, level_cell_array_2);
    // set.apply([](auto& index_yz, auto& interval){std::cout << index_yz << " " << interval << "\n";});

    /////////////////////////////////////////
    //
    // 3D
    //
    /////////////////////////////////////////
    // constexpr std::size_t dim = 3;
    // using config = samurai::MRConfig<dim>;

    // samurai::LevelCellList<config> level_cell_list_1;
    // level_cell_list_1.extend({0, 0}, {4, 4});
    // level_cell_list_1[{0, 0}].add_interval({0, 2});
    // // level_cell_list_1[0].add_interval({4, 7});
    // samurai::LevelCellArray<config> level_cell_array_1{level_cell_list_1};

    // samurai::LevelCellList<config> level_cell_list_2;
    // level_cell_list_2.extend({0, 0}, {4, 4});
    // level_cell_list_2[{0, 0}].add_interval({0, 5});
    // level_cell_list_2[{0, 1}].add_interval({0, 2});

    // samurai::LevelCellArray<config> level_cell_array_2{level_cell_list_2};

    // auto expr = samurai::intersection(samurai::_2, samurai::_1);
    // auto level_cell_arrays = std::tie(level_cell_array_1,
    //                                   level_cell_array_2);

    // auto set = samurai::make_subset<config>(expr, level_cell_array_1, level_cell_array_2);
    // set.apply([](auto& index_yz, auto& interval){std::cout << index_yz << " " << interval << "\n";});
}