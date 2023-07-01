// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <filesystem>
namespace fs = std::filesystem;

#include <samurai/hdf5.hpp>
#include <samurai/io/from_geometry.hpp>

template <class Mesh>
void save_mesh(const fs::path& path, const std::string& filename, const Mesh& mesh)
{
    auto level = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               level[cell] = cell.level;
                           });

    samurai::save(path, filename, mesh, level);
}

int main(int argc, char** argv)
{
    constexpr std::size_t dim = 3;
    std::size_t start_level   = 1;
    std::size_t max_level     = 8;
    bool keep_inside          = false;
    bool keep_outside         = false;

    // Output parameters
    fs::path path = fs::current_path();
    std::string input_file;

    CLI::App app{"Create an adapted mesh from an OBJ file"};
    app.add_option("--input", input_file, "input File")->required()->check(CLI::ExistingFile);
    app.add_option("--start-level", start_level, "Start level of the output adaptive mesh")->capture_default_str();
    app.add_flag("--keep-inside", keep_inside, "Keep the cells inside the object")->capture_default_str();
    app.add_flag("--keep-outside", keep_outside, "Keep the cells outside the object")->capture_default_str();
    app.add_option("--max-level", max_level, "Maximum level of the output adaptive mesh")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    std::string output_file = fs::path(input_file).stem();

    auto mesh = samurai::from_geometry<dim>(input_file, start_level, max_level, keep_outside, keep_inside);
    make_graduation(mesh);

    save_mesh(path, fmt::format("mesh_{}", output_file), mesh);

    return EXIT_SUCCESS;
}
