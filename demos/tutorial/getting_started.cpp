// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Program of the getting started tutorial (docs/source/tutorial/getting_started.md):
// advection of a disc on a 2D mesh adapted by multiresolution.

#include <array>
#include <iostream>
#include <string>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>

// Save the solution together with the level of each cell
template <class Field>
void save(const std::string& filename, Field& u)
{
    auto& mesh = u.mesh();
    auto level = samurai::make_scalar_field<std::size_t>("level", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               level[cell] = cell.level;
                           });
    samurai::save(filename, mesh, u, level);
}

int main(int argc, char* argv[])
{
    samurai::initialize("Getting started: advection of a disc on an adaptive mesh", argc, argv);
    SAMURAI_PARSE(argc, argv);

    constexpr std::size_t dim = 2;

    // Create the mesh
    const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>().min_level(4).max_level(8);
    auto mesh   = samurai::mra::make_mesh(box, config);

    // Create the field
    auto u = samurai::make_scalar_field<double>("u", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const auto x        = cell.center();
                               const double radius = 0.2;
                               const double dx     = x[0] - 0.3;
                               const double dy     = x[1] - 0.3;
                               u[cell]             = (dx * dx + dy * dy <= radius * radius) ? 1. : 0.;
                           });
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    // Adapt the mesh
    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(2e-4);
    MRadaptation(mra_config);
    save("getting_started_init", u);

    // Solve the advection equation
    const std::array<double, dim> velocity{1., 1.};
    const double Tf  = 0.3;
    const double cfl = 0.5;
    double dt        = cfl * mesh.min_cell_length();
    double t         = 0.;
    std::size_t nt   = 0;

    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    while (t != Tf)
    {
        MRadaptation(mra_config);

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.6f}, dt = {:.6f}", nt++, t, dt) << std::endl;

        samurai::update_ghost_mr(u);
        unp1.resize();
        unp1 = u - dt * samurai::upwind(velocity, u);

        std::swap(u.array(), unp1.array());
    }
    save("getting_started", u);

    samurai::finalize();
    return 0;
}
