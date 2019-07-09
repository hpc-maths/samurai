#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>
#include <mure/mesh.hpp>
#include <mure/mr_config.hpp>
#include <mure/level_cell_list.hpp>
#include <mure/level_cell_array.hpp>

int main()
{
    // constexpr size_t dim = 2;
    // using Config = mure::MRConfig<dim>;

    // mure::CellList<Config> cell_list;
    // cell_list[2][{0}].add_interval({0, 1});
    // cell_list[2][{0}].add_interval({3, 4});
    // cell_list[2][{3}].add_interval({0, 1});
    // cell_list[2][{3}].add_interval({3, 4});
    
    // cell_list[3][{0}].add_interval({2, 6});
    // cell_list[3][{1}].add_interval({2, 6});
    // cell_list[3][{2}].add_interval({0, 3});
    // cell_list[3][{2}].add_interval({5, 8});
    // cell_list[3][{3}].add_interval({0, 2});
    // cell_list[3][{3}].add_interval({3, 5});
    // cell_list[3][{3}].add_interval({6, 8});
    // cell_list[3][{4}].add_interval({0, 2});
    // cell_list[3][{4}].add_interval({3, 5});
    // cell_list[3][{4}].add_interval({6, 8});
    // cell_list[3][{5}].add_interval({0, 3});
    // cell_list[3][{5}].add_interval({5, 8});
    // cell_list[3][{6}].add_interval({2, 6});
    // cell_list[3][{7}].add_interval({2, 6});

    // cell_list[4][{4}].add_interval({6, 10});
    // cell_list[4][{5}].add_interval({6, 10});
    // cell_list[4][{6}].add_interval({4, 6});
    // cell_list[4][{6}].add_interval({10, 12});
    // cell_list[4][{7}].add_interval({4, 6});
    // cell_list[4][{7}].add_interval({10, 12});
    // cell_list[4][{8}].add_interval({4, 6});
    // cell_list[4][{8}].add_interval({10, 12});
    // cell_list[4][{9}].add_interval({4, 6});
    // cell_list[4][{9}].add_interval({10, 12});
    // cell_list[4][{10}].add_interval({6, 10});
    // cell_list[4][{11}].add_interval({6, 10});

    constexpr size_t dim = 1;
    using Config = mure::MRConfig<dim>;

    mure::CellList<Config> cell_list;
    cell_list[0][{}].add_point(0);
    cell_list[0][{}].add_point(4);
    
    cell_list[1][{}].add_point(2);
    cell_list[1][{}].add_interval({6, 8});

    cell_list[2][{}].add_interval({6, 12});
    mure::Mesh<Config> mesh{cell_list};

    std::cout << mesh << "\n";

    // // projection
    // for(std::size_t level=2; level>0; --level)
    // {
    //     auto expr = mure::intersection(mure::_1, mure::_2);
    //     auto set = mure::make_subset<Config>(expr,
    //                                            level-1,
    //                                            {level-1, level},
    //                                             mesh.m_all_cells[level-1],
    //                                             mesh.m_cells[level]);

    //     set.apply([&](auto& index_yz, auto& interval, auto& interval_index)
    //     {
    //         std::cout << level << " " << interval << "\n";
    //     });
    // }

    // projection
    for(std::size_t level=3; level>0; --level)
    {
        auto expr = mure::union_(mure::intersection(mure::_1, mure::_2), mure::_3);
        auto set = mure::make_subset<Config>(expr,
                                                  level-1,
                                                  {level, level+1, level},
                                                  mesh.m_all_cells[level],
                                                  mesh.m_all_cells[level+1],
                                                  mesh.m_cells[level]);

        set.apply([&](auto& index_yz, auto& interval, auto& interval_index)
        {
            std::cout << level << " " << interval << "\n";
        });
    }

    // auto h5file = mure::Hdf5("mesh");
    // h5file.add_mesh(mesh);
    return 0;
}