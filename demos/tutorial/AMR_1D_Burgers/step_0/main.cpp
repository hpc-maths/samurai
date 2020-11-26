#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/hdf5.hpp>

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t init_level = 4;

    samurai::Box<double, dim> box({-2}, {2});
    samurai::CellArray<dim> mesh;

    mesh[init_level] = {init_level, box};

    std::cout << mesh << "\n";

    samurai::save("Step0", mesh);

    return 0;
}