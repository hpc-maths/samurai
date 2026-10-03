#include <cmath>
#include <iostream>

#include <samurai/algorithm.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

int main(int argc, char* argv[])
{
    // Read the options
    auto& app = samurai::initialize("Burgers equation on a uniform 1D mesh", argc, argv);

    double Tf  = 1.5;  // final time, after the shock forms at T* = 1
    double cfl = 0.99; // CFL number: dt = cfl * dx, stable for cfl <= 1
    app.add_option("--Tf", Tf, "Final time")->capture_default_str();
    app.add_option("--cfl", cfl, "CFL number")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    // Build the mesh
    constexpr std::size_t dim = 1;
    const std::size_t level   = 8; // J: 2^8 = 256 cells

    const samurai::Box<double, dim> box({-3}, {3});
    auto config = samurai::mesh_config<dim>().min_level(level).max_level(level);
    auto mesh   = samurai::amr::make_mesh(box, config);

    std::cout << mesh << std::endl;

    // Set the initial condition
    auto u = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               const double x = cell.center(0);
                               u[cell]        = (x < -1. || x > 1.) ? 0. : 1. - std::abs(x);
                           });

    // Set the boundary condition
    samurai::make_bc<samurai::Neumann<1>>(u, 0.);

    // Define the scheme
    auto conv = 0.5 * samurai::make_convection_upwind<decltype(u)>();

    // Write the time loop
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    double dt      = cfl * mesh.min_cell_length(); // sup |u_0| = 1
    double t       = 0.;
    std::size_t nt = 0;

    while (t != Tf)
    {
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        unp1 = u - dt * conv(u);

        samurai::swap(u, unp1);

        std::cout << "iteration " << ++nt << ": t = " << t << std::endl;
    }

    // Save the solution
    samurai::save("burgers_1d", mesh, u);

    samurai::finalize();
    return 0;
}
