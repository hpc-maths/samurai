#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t init_level = 8;

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

    double Tf = 1.5; // We have blowup at t = 1
    double dx = 1./(1 << init_level);
    double dt = 0.99 * dx; // 0.99 * dx

    double t = 0.;
    std::size_t it = 0;

    auto phi_np1 = samurai::make_field<double, 1>("phi", mesh);

    while (t < Tf)
    {
        fmt::print("Iteration = {:4d} Time = {:5.4}\n", it, t);

        samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, auto)
        {
            using interval_t = decltype(interval);

            auto new_i = interval_t{interval.start + 1, interval.end - 1};
            phi_np1(level, new_i) = phi(level, new_i) - .5*dt/dx*(xt::pow(phi(level, new_i), 2.) - xt::pow(phi(level, new_i - 1), 2.));
        });

        std::swap(phi.array(), phi_np1.array());
        t += dt;

        samurai::save(fmt::format("Step2_ite-{}", it++), mesh, phi);
    }

    return 0;
}
