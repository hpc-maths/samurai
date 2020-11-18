#include <cstddef>
#include <iostream>

#include <samurai/mr_config.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/level_cell_array.hpp>

#include <xtensor/xview.hpp>

int main()
{
    constexpr std::size_t dim = 3;
    using Config = samurai::MRConfig<dim>;
    using coord_index_t = Config::coord_index_t;
    const coord_index_t cross_size = 5;

    // Creating the level cell list
    samurai::LevelCellList<Config> dcl;

    Config::index_t cnt = 0;
    for (Config::coord_index_t i = 0; i < cross_size; ++i)
    {
        if (i != cross_size / 2)
        {
            dcl[{i,i}].add_interval({i, i+1, cnt++});
            dcl[{i,i}].add_interval({cross_size-i-1, cross_size-i, cnt++});
            dcl[{cross_size-i-1,i}].add_interval({i, i+1, cnt++});
            dcl[{cross_size-i-1,i}].add_interval({cross_size-i-1, cross_size-i, cnt++});
        }
    }

    // Converting it to a level cell array
    std::cout << "The level cell array:" << std::endl;
    samurai::LevelCellArray<Config> dca(dcl);

    std::cout << dca << std::endl;

    // Visiting it
    std::cout << "Visiting it:" << std::endl;
    dca.for_each_interval_in_x([] (auto index, auto interval) { std::cout << index << "x" << interval << std::endl; });

    return 0;
}
