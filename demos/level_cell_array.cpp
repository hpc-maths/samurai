#include <cstddef>
#include <iostream>

#include <mure/box.hpp>
#include <mure/mr_config.hpp>
#include <mure/level_cell_list.hpp>
//#include <mure/level_cell_array_other.hpp>
//#include <mure/level_cell_array_other_nodeque.hpp>
#include <mure/level_cell_array.hpp>

#include <xtensor/xview.hpp>

int main()
{
    constexpr std::size_t dim = 3;
    using Config = mure::MRConfig<dim>;
    using coord_index_t = Config::coord_index_t;
    const coord_index_t cross_size = 5;

    // Creating the box
    using box_t = mure::Box<coord_index_t, dim>;
    box_t::point_t min_corner; min_corner.fill(0);
    box_t::point_t max_corner; max_corner.fill(cross_size);
    box_t box(min_corner, max_corner);

    // Creating the level cell list
    mure::LevelCellList<Config> dcl;
    dcl.extend(xt::view(box.min_corner(), xt::drop(0)), xt::view(box.max_corner(), xt::drop(0)));

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
    mure::LevelCellArray<Config> dca(dcl);

    std::cout << dca << std::endl;

    // Visiting it
    std::cout << "Visiting it:" << std::endl;
    dca.for_each_interval_in_x([] (auto index, auto interval) { std::cout << index << "x" << interval << std::endl; });

    return 0;
}
