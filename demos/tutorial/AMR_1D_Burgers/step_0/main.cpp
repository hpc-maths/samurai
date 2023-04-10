// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <filesystem>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/hdf5.hpp>

namespace fs = std::filesystem;

/**
 * What will we learn ?
 * ====================
 *
 * - construct 1D uniform grid from a Box
 * - print mesh information
 * - save and plot a mesh
 *
 */

int main(int argc, char* argv[])
{
    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "amr_1d_burgers_step_0";

    CLI::App app{"Tutorial AMR Burgers 1D step 0"};
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    constexpr std::size_t dim    = 1;
    const std::size_t init_level = 4;

    /**
     *
     * level: 1   |--|--|--|--|--|--|--|--|
     *
     * level: 0   |-----|-----|-----|-----|
     *           -2    -1     0     1     2
     */

    const samurai::Box<double, dim> box({-2}, {2});
    samurai::CellArray<dim> mesh;

    mesh[init_level] = {init_level, box};

    std::cout << mesh << "\n";

    samurai::save(path, filename, mesh);

    return 0;
}
