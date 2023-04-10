#include <iostream>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

int main()
{
    constexpr std::size_t dim = 1;
    samurai::CellList<dim> cl;

    cl[0][{}].add_interval({0, 2});
    cl[0][{}].add_interval({5, 6});
    cl[1][{}].add_interval({4, 7});
    cl[1][{}].add_interval({8, 10});
    cl[2][{}].add_interval({14, 16});

    samurai::CellArray<dim> ca{cl};

    std::cout << ca << std::endl;

    return 0;
}
