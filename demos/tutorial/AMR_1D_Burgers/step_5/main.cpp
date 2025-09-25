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
    auto& app = samurai::initialize("Tutorial AMR Burgers 1D step 5", argc, argv);

    // AMR parameters
    std::size_t start_level = 8;
    std::size_t min_level   = 2;
    std::size_t max_level   = 8;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "amr_1d_burgers_step_5";

    app.add_option("--start-level", start_level, "Start level of the AMR")->capture_default_str()->group("AMR parameter");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
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

    std::size_t i_adapt = 0;
    while (i_adapt < (max_level - min_level + 1))
    {
        auto tag = samurai::make_scalar_field<std::size_t>("tag", mesh);

        update_ghost(phi); // <--------------------------------
        AMR_criterion(phi, tag);
        make_graduation(tag); // <--------------------------------

        samurai::save(path, fmt::format("{}_criterion-{}", filename, i_adapt++), mesh, phi, tag);

        if (update_mesh(phi, tag))
        {
            break;
        };
    }

    auto level = samurai::make_scalar_field<std::size_t>("level", mesh);
    samurai::for_each_interval(mesh[MeshID::cells],
                               [&](std::size_t lvl, const auto& i, auto)
                               {
                                   level(lvl, i) = lvl;
                               });
    samurai::save(path, filename, mesh, phi, level);

    samurai::finalize();
    return 0;
}
