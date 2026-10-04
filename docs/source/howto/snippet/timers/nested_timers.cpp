#include <cmath>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>

int main(int argc, char** argv)
{
    static constexpr std::size_t dim = 2;

    samurai::initialize("Nested timers example", argc, argv);

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(10);
    auto mesh = samurai::mra::make_mesh(box, config);

    using mesh_id_t            = typename decltype(mesh)::mesh_id_t;
    const auto nb_cells        = mesh.nb_cells(mesh_id_t::cells);
    const double dt            = 0.01;
    const std::size_t nb_steps = 100;

    auto u   = samurai::make_scalar_field<double>("u", mesh);
    auto rhs = samurai::make_scalar_field<double>("rhs", mesh);

    // Solve du/dt = u (1 - u) with the explicit Euler method
    auto init = [&](const auto& cell)
    {
        const double x = cell.center(0) - 0.5;
        const double y = cell.center(1) - 0.5;
        u[cell]        = std::exp(-20. * (x * x + y * y));
    };
    auto compute_rhs = [&](const auto& cell)
    {
        rhs[cell] = u[cell] * (1. - u[cell]);
    };
    auto update = [&](const auto& cell)
    {
        u[cell] += dt * rhs[cell];
    };

    {
        samurai::ScopedTimer init_timer("init field");
        samurai::for_each_cell(mesh, init);
    }

    {
        samurai::ScopedTimer loop_timer("time loop");
        for (std::size_t nt = 0; nt < nb_steps; ++nt)
        {
            {
                samurai::ScopedTimer rhs_timer("rhs computation");
                samurai::for_each_cell(mesh, compute_rhs);
                rhs_timer.set_cells(nb_cells);
            }

            samurai::times::timers.start("update");
            samurai::for_each_cell(mesh, update);
            samurai::times::timers.stop("update", nb_cells);
        }
    }

    samurai::finalize();
    return 0;
}
