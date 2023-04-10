// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <filesystem>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "init_sol.hpp"

namespace fs = std::filesystem;

/**
 * What will we learn ?
 * ====================
 *
 * - create a field
 * - initialize this field
 * - save and plot a field
 *
 */

int main(int argc, char* argv[])
{
    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "amr_1d_burgers_step_1";

    CLI::App app{"Tutorial AMR Burgers 1D step 1"};
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    constexpr std::size_t dim    = 1;
    const std::size_t init_level = 6;

    const samurai::Box<double, dim> box({-3}, {3});
    samurai::CellArray<dim> mesh;

    mesh[init_level] = {init_level, box};

    //////////////////////////////////
    auto phi = init_sol(mesh);
    /////////////////////////////////

    std::cout << mesh << "\n";

    samurai::save(path, filename, mesh, phi);

    return 0;
}
