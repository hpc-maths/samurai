#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t init_level = 4;

    samurai::Box<double, dim> box({-2}, {2});
    samurai::CellArray<dim> mesh;

    mesh[init_level] = {init_level, box};

    auto phi = samurai::make_field<double, 1>("phi", mesh);

    samurai::for_each_cell(mesh, [&](auto &cell)
    {
        double x = cell.center(0);

        // Initial hat solution
        if (x < -1. or x > 1.)
        {
            phi[cell] = 0.;
        }
        else
        {
            phi[cell] = (x < 0.) ? (1 + x) : (1 - x);
        }
    });

    std::cout << mesh << "\n";

    samurai::save("Step1", mesh, phi);

    return 0;
}