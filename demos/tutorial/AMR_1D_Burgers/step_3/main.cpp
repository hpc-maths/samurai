// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <filesystem>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/samurai.hpp>

#include "init_sol.hpp"
#include "mesh.hpp"
#include "update_sol.hpp"

namespace fs = std::filesystem;

/**
 * What will we learn ?
 * ====================
 *
 * - create a new mesh data structure with 2 CellArray
 * - how to modify init_sol and update_sol accordingly
 * - save and plot a field on this mesh
 *
 */

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Tutorial AMR Burgers 1D step 3", argc, argv);

    // Simulation parameters
    double cfl = 0.99;
    double Tf  = 1.5;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "amr_1d_burgers_step_3";
    std::size_t nfiles   = 1;

    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    constexpr std::size_t dim     = 1;
    const std::size_t start_level = 6; // <--------------------------------
    const std::size_t min_level   = 6; // <--------------------------------
    const std::size_t max_level   = 6; // <--------------------------------

    const samurai::Box<double, dim> box({-3}, {3});
    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
    Mesh<MeshConfig<dim>> mesh(config, box, start_level); // <--------------------------------

    auto phi = init_sol(mesh);

    const double dx      = mesh.cell_length(max_level);
    double dt            = cfl * dx;
    const double dt_save = Tf / static_cast<double>(nfiles);

    double t          = 0.;
    std::size_t nsave = 1;
    std::size_t nt    = 0;

    auto phi_np1 = samurai::make_scalar_field<double>("phi", mesh);
    phi_np1.fill(0.);

    while (t != Tf)
    {
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        fmt::print("Iteration = {:4d} Time = {:5.4}\n", nt++, t);

        update_sol(dt, phi, phi_np1);

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, phi);
        }
    }

    samurai::finalize();
    return 0;
}
