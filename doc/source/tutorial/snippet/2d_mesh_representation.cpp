#include <iostream>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/hdf5.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    samurai::CellList<dim> cl;

    cl[0][{0}].add_interval({0, 4});
    cl[0][{1}].add_interval({0, 1});
    cl[0][{1}].add_interval({3, 4});
    cl[0][{2}].add_interval({0, 1});
    cl[0][{2}].add_interval({3, 4});
    cl[0][{3}].add_interval({0, 3});

    cl[1][{2}].add_interval({2, 6});
    cl[1][{3}].add_interval({2, 6});
    cl[1][{4}].add_interval({2, 4});
    cl[1][{4}].add_interval({5, 6});
    cl[1][{5}].add_interval({2, 6});
    cl[1][{6}].add_interval({6, 8});
    cl[1][{7}].add_interval({6, 7});

    cl[2][{8}].add_interval({8, 10});
    cl[2][{8}].add_interval({8, 10});
    cl[2][{14}].add_interval({14, 16});
    cl[2][{15}].add_interval({14, 16});

    samurai::CellArray<dim> ca{cl};

    std::cout << ca << std::endl;

    samurai::save("2d_mesh_representation", ca);
    return 0;
}
