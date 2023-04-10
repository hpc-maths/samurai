// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <CLI/CLI.hpp>
#include <iostream>

#include <filesystem>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/hdf5.hpp>

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 2;
    samurai::CellList<dim> cl;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "2d_mesh_construction";

    CLI::App app{"Create mesh from CellList and save it"};
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    cl[0][{0}].add_interval({0, 4});
    cl[0][{1}].add_interval({0, 1});
    cl[0][{1}].add_interval({3, 4});
    cl[0][{2}].add_interval({0, 1});
    cl[0][{2}].add_interval({3, 4});
    cl[0][{3}].add_interval({0, 3});

    cl[1][{2}].add_interval({2, 6});
    cl[1][{3}].add_interval({2, 6});
    cl[1][{4}].add_interval({2, 4});
    cl[1][{4}].add_interval({5, 6});
    cl[1][{5}].add_interval({2, 6});
    cl[1][{6}].add_interval({6, 8});
    cl[1][{7}].add_interval({6, 7});

    cl[2][{8}].add_interval({8, 10});
    cl[2][{9}].add_interval({8, 10});
    cl[2][{14}].add_interval({14, 16});
    cl[2][{15}].add_interval({14, 16});

    const samurai::CellArray<dim> ca{cl};

    std::cout << ca << std::endl;

    samurai::save(path, filename, ca);
    return 0;
}
