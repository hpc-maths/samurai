
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "../step_3/init_sol.hpp"
#include "../step_3/mesh.hpp"
// #include "../step_3/update_sol.hpp" // without flux correction

#include "../step_4/AMR_criterion.hpp"
#include "../step_4/update_mesh.hpp"

#include "../step_5/update_ghost.hpp"
#include "../step_5/make_graduation.hpp"

#include "update_sol.hpp" // with flux correction

/**
 * What will we learn ?
 * ====================
 *
 * - add the time loop
 * - adapt the mesh at each time iteration
 * - solve the 1D Burgers equation on the adapted mesh
 *
 */

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t start_level = 8;
    std::size_t min_level = 2;
    std::size_t max_level = 8;

    samurai::Box<double, dim> box({-3}, {3});
    Mesh<MeshConfig<dim>> mesh(box, start_level, min_level, max_level);

    auto phi = init_sol(mesh);

    double Tf = 1.5;
    double dx = 1./(1 << max_level);
    double dt = 0.99 * dx;

    double t = 0.;
    std::size_t it = 0;

    while (t < Tf)
    {
        fmt::print("Iteration = {:4d} Time = {:5.4}\n", it, t);

        std::size_t i_adapt = 0;
        while(i_adapt < (max_level - min_level + 1))
            {
            auto tag = samurai::make_field<std::size_t, 1>("tag", mesh);

            fmt::print("adaptation iteration : {:4d}\n", i_adapt++);
            update_ghost(phi);
            AMR_criterion(phi, tag);
            make_graduation(tag);

            if (update_mesh(phi, tag))
            {
                break;
            };
        }

        update_ghost(phi);
        auto phi_np1 = samurai::make_field<double, 1>("phi", mesh);

        update_sol(dt, phi, phi_np1);
        t += dt;

        auto level = samurai::make_field<int, 1>("level", mesh);
        samurai::for_each_interval(mesh[MeshID::cells], [&](std::size_t l, const auto& i, auto)
        {
            level(l, i) = l;
        });
        samurai::save(fmt::format("step_6-{}", it++), mesh, level, phi);
    }

    return 0;
}

