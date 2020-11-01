#include <iostream>

#include <mure/cell_array.hpp>
#include <mure/cell_list.hpp>
#include <mure/hdf5.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    mure::CellList<dim> cl;

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

    cl[2][{8}].add_interval({ 8, 10});
    cl[2][{9}].add_interval({ 8, 10});
    cl[2][{14}].add_interval({14, 16});
    cl[2][{15}].add_interval({14, 16});

    mure::CellArray<dim> ca{cl};

    std::cout << ca << std::endl;

    mure::save("test", ca);
    return 0;
}