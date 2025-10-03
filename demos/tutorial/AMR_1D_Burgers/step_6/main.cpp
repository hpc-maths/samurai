// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <filesystem>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/samurai.hpp>

#include "../step_3/init_sol.hpp"
#include "../step_3/mesh.hpp"
// #include "../step_3/update_sol.hpp" // without flux correction

#include "../step_4/AMR_criterion.hpp"
#include "../step_4/update_mesh.hpp"

#include "../step_5/make_graduation.hpp"
#include "../step_5/update_ghost.hpp"

#include "update_sol.hpp" // with flux correction

namespace fs = std::filesystem;

/**
 * What will we learn ?
 * ====================
 *
 * - add the time loop
 * - adapt the mesh at each time iteration
 * - solve the 1D Burgers equation on the adapted mesh
 *
 */

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Tutorial AMR Burgers 1D step 6", argc, argv);

    // Simulation parameters
    double cfl         = 0.99;
    double Tf          = 1.5;
    std::size_t nfiles = 1;

    // AMR parameters
    std::size_t start_level = 8;
    std::size_t min_level   = 2;
    std::size_t max_level   = 8;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "amr_1d_burgers_step_6";

    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--start-level", start_level, "Start level of the AMR")->capture_default_str()->group("AMR parameter");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    constexpr std::size_t dim = 1;

    const samurai::Box<double, dim> box({-3}, {3});
    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
    Mesh<MeshConfig<dim>> mesh(config, box, start_level);

    auto phi = init_sol(mesh);

    const double dx      = mesh.cell_length(max_level);
    double dt            = cfl * dx;
    const double dt_save = Tf / static_cast<double>(nfiles);

    double t          = 0.;
    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        fmt::print("Iteration = {:4d} Time = {:5.4}\n", nt++, t);

        std::size_t i_adapt = 0;
        while (i_adapt < (max_level - min_level + 1))
        {
            auto tag = samurai::make_scalar_field<std::size_t>("tag", mesh);

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
        auto phi_np1 = samurai::make_scalar_field<double>("phi", mesh);

        update_sol(dt, phi, phi_np1);

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            auto level = samurai::make_scalar_field<std::size_t>("level", mesh);
            samurai::for_each_interval(mesh[MeshID::cells],
                                       [&](std::size_t lvl, const auto& i, auto)
                                       {
                                           level(lvl, i) = lvl;
                                       });

            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, phi, level);
        }
    }

    samurai::finalize();
    return 0;
}
