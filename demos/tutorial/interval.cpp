#include <iostream>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

int main()
{
    constexpr std::size_t dim = 2;

    samurai::CellList<dim> cl;

    cl[0][{}].add_interval({0, 2});
    cl[0][{}].add_interval({5, 6});
    cl[1][{}].add_interval({4, 7});
    cl[1][{}].add_interval({8, 10});
    cl[2][{}].add_interval({15, 17});

    samurai::CellArray<dim> ca{cl};

    std::cout << ca << std::endl;

    constexpr std::size_t start_level = 3;
    samurai::Box<double, dim> box({-1, -1}, {1, 1});
    samurai::CellArray<dim> ca_box;

    ca_box[start_level] = {start_level, box};

    std::cout << ca_box << std::endl;

}