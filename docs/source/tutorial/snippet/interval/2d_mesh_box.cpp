#include <iostream>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize("2D mesh built from a box", argc, argv);
    SAMURAI_PARSE(argc, argv);

    constexpr std::size_t dim   = 2;
    constexpr std::size_t level = 3;

    const samurai::Box<double, dim> box({-1, -1}, {1, 1});
    samurai::CellArray<dim> ca;

    ca[level] = {level, box};

    std::cout << ca << std::endl;

    samurai::finalize();
    return 0;
}
