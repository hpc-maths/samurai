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

#include "../step_4/AMR_criterion.hpp"
#include "../step_4/update_mesh.hpp"

#include "make_graduation.hpp"
#include "update_ghost.hpp"

namespace fs = std::filesystem;

/**
 * What will we learn ?
 * ====================
 *
 * - update the ghost cells
 * - make the graduation of the mesh
 *
 */

int main(int argc, char* argv[])
{
    // AMR parameters
    std::size_t start_level = 8;
    std::size_t min_level   = 2;
    std::size_t max_level   = 8;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "amr_1d_burgers_step_5";

    CLI::App app{"Tutorial AMR Burgers 1D step 5"};
    app.add_option("--start-level", start_level, "Start level of the AMR")->capture_default_str()->group("AMR parameter");
    app.add_option("--min-level", min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
    app.add_option("--max-level", max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    constexpr std::size_t dim = 1;

    const samurai::Box<double, dim> box({-3}, {3});
    Mesh<MeshConfig<dim>> mesh(box, start_level, min_level, max_level);

    auto phi = init_sol(mesh);

    std::size_t i_adapt = 0;
    while (i_adapt < (max_level - min_level + 1))
    {
        auto tag = samurai::make_field<std::size_t, 1>("tag", mesh);

        update_ghost(phi); // <--------------------------------
        AMR_criterion(phi, tag);
        make_graduation(tag); // <--------------------------------

        samurai::save(path, fmt::format("{}_criterion-{}", filename, i_adapt++), mesh, phi, tag);

        if (update_mesh(phi, tag))
        {
            break;
        };
    }

    auto level = samurai::make_field<std::size_t, 1>("level", mesh);
    samurai::for_each_interval(mesh[MeshID::cells],
                               [&](std::size_t lvl, const auto& i, auto)
                               {
                                   level(lvl, i) = lvl;
                               });
    samurai::save(path, filename, mesh, phi, level);

    return 0;
}
