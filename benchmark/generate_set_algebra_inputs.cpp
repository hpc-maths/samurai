// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Adapts the meshes of bench::mesh_cases, unmoved and moved, and writes their
// sets to the files bench_set_call_sites and bench_dynamic_set read (format
// in site_inputs.hpp). Each file is read back and compared with the mesh it
// comes from.

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "mr_fixtures.hpp"

namespace
{
    template <std::size_t dim>
    void generate()
    {
        for (const auto& args : bench::mesh_cases<dim>())
        {
            const auto max_level = static_cast<std::size_t>(args[0]);
            const auto nballs    = static_cast<std::size_t>(args[1]);
            for (const bool moved : {false, true})
            {
                const auto in   = bench::make_site_inputs<dim>(bench::adapted_case<dim>(max_level, nballs, moved).mesh);
                const auto path = bench::inputs_file<dim>(max_level, nballs, moved);
                bench::write_site_inputs(in, path);
                if (!(bench::read_site_inputs<dim>(path) == in))
                {
                    throw std::runtime_error(path.string() + " does not read back as the mesh it was written from");
                }
                std::cout << path.string() << ": " << in.cells.nb_cells() << " leaves\n";
            }
        }
    }
} // namespace

int main()
{
    try
    {
        std::filesystem::create_directories(bench::inputs_dir());
        generate<1>();
        generate<2>();
        generate<3>();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
