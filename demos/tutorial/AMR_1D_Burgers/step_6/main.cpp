// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <filesystem>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

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

    CLI::App app{"Tutorial AMR Burgers 1D step 6"};
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--start-level", start_level, "Start level of the AMR")->capture_default_str()->group("AMR parameter");
    app.add_option("--min-level", min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
    app.add_option("--max-level", max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    constexpr std::size_t dim = 1;

    const samurai::Box<double, dim> box({-3}, {3});
    Mesh<MeshConfig<dim>> mesh(box, start_level, min_level, max_level);

    auto phi = init_sol(mesh);

    const double dx      = 1. / (1 << max_level);
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

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            auto level = samurai::make_field<std::size_t, 1>("level", mesh);
            samurai::for_each_interval(mesh[MeshID::cells],
                                       [&](std::size_t lvl, const auto& i, auto)
                                       {
                                           level(lvl, i) = lvl;
                                       });

            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, phi, level);
        }
    }

    return 0;
}
