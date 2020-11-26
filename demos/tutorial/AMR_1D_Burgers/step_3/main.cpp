#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "mesh.hpp"
#include "init_sol.hpp"
#include "update_sol.hpp"

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t start_level = 6;                                          // <--------------------------------
    std::size_t min_level = 6;                                            // <--------------------------------
    std::size_t max_level = 6;                                            // <--------------------------------

    samurai::Box<double, dim> box({-3}, {3});
    Mesh<MeshConfig<dim>> mesh(box, start_level, min_level, max_level);   // <--------------------------------

    auto phi = init_sol(mesh);

    double Tf = 1.5;
    double dx = 1./(1 << max_level);
    double dt = 0.99 * dx;

    double t = 0.;
    std::size_t it = 0;

    auto phi_np1 = samurai::make_field<double, 1>("phi", mesh);

    while (t < Tf)
    {
        fmt::print("Iteration = {:4d} Time = {:5.4}\n", it, t);

        update_sol(dt, phi, phi_np1);

        t += dt;

        samurai::save(fmt::format("step3-{}", it++), mesh, phi);
    }

    return 0;
}
