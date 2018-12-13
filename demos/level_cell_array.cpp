#include <cstddef>
#include <iostream>

#include <mure/box.hpp>
#include <mure/mr_config.hpp>
#include <mure/level_cell_list.hpp>
//#include <mure/level_cell_array_other.hpp>
//#include <mure/level_cell_array_other_nolist.hpp>
#include <mure/level_cell_array.hpp>

#include <xtensor/xview.hpp>

int main()
{
    constexpr std::size_t dim = 3;
    using Config = mure::MRConfig<dim>;
    using coord_index_t = Config::coord_index_t;
    const coord_index_t cross_size = 5;

    mure::Box<coord_index_t, dim> box{{0, 0, 0}, {cross_size, cross_size, cross_size}};
    mure::LevelCellList<Config> dcl;
    dcl.extend(xt::view(box.min_corner(), xt::drop(0)), xt::view(box.max_corner(), xt::drop(0)));

    std::size_t cnt = 0;
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

    mure::LevelCellArray<Config> dca(dcl);

    std::cout << dca << std::endl;

    dca.for_each_interval_in_x([] (auto index, auto interval) { std::cout << index << "x" << interval << std::endl; });

    return 0;
}
