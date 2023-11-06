#include <iostream>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/hdf5.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    std::size_t start_level   = 3;

    samurai::Box<double, dim> box({-1, -1}, {1, 1});
    samurai::CellArray<dim> ca_box;

    ca_box[start_level] = {start_level, box};

    std::cout << ca_box << std::endl;

    samurai::save("2d_mesh_box", ca_box);
    return 0;
}
