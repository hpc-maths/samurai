// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/cell_array.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/samurai.hpp>

#include <filesystem>
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
    auto& app = samurai::initialize("Tutorial AMR Burgers 1D step 0", argc, argv);

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "amr_1d_burgers_step_0";

    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    constexpr std::size_t dim    = 1; // cppcheck-suppress unreadVariable
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

    mesh[init_level] = {init_level, box, 0, 1};

    std::cout << mesh << "\n";

    samurai::save(path, filename, mesh);

    samurai::finalize();
    return 0;
}
