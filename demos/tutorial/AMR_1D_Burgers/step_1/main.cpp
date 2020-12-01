#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "init_sol.hpp"

/**
 * What will we learn ?
 * ====================
 *
 * - create a field
 * - initialize this field
 * - save and plot a field
 *
 */

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t init_level = 6;

    samurai::Box<double, dim> box({-3}, {3});
    samurai::CellArray<dim> mesh;

    mesh[init_level] = {init_level, box};

    //////////////////////////////////
    auto phi = init_sol(mesh);
    /////////////////////////////////

    std::cout << mesh << "\n";

    samurai::save("step_1", mesh, phi);

    return 0;
}