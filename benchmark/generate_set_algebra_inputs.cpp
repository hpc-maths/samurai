// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Adapts the meshes of bench::mesh_cases, unmoved and moved, and writes their
// sets, and the cells of the next step before graduation with the sets
// update_cell_array_from_tag builds them from, to the files
// bench_set_call_sites, bench_dynamic_set and bench_set_graduation read
// (format in site_inputs.hpp). Then writes the partitions of
// bench::partition_cases, the meshes of their ranks. Each file is read back
// and compared with what it was written from.

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
        for (const auto& c : bench::mesh_cases<dim>())
        {
            for (const bool moved : {false, true})
            {
                const auto& mesh = bench::adapted_case<dim>(c, moved).mesh;
                auto in          = bench::make_site_inputs<dim>(mesh);
                const auto shift = (moved ? 2. : 1.) * bench::step_shift(mesh);
                bench::add_next_step<dim>(in, mesh, c.nballs, shift);
                const auto path = bench::inputs_file<dim>(c, moved);
                bench::write_site_inputs(in, path);
                if (!(bench::read_site_inputs<dim>(path) == in))
                {
                    throw std::runtime_error(path.string() + " does not read back as the mesh it was written from");
                }
                std::cout << path.string() << ": " << in.cells.nb_cells() << " leaves\n";
            }
        }
        for (const auto& c : bench::partition_cases<dim>())
        {
            const auto ranks = bench::make_partition<dim>(c);
            const auto path  = bench::partition_file<dim>(c);
            bench::write_partition(ranks, path);
            if (!(bench::read_partition<dim>(path) == ranks))
            {
                throw std::runtime_error(path.string() + " does not read back as the partition it was written from");
            }
            std::size_t pairs = 0;
            for (const auto& in : ranks)
            {
                pairs += in.neighbours.size();
            }
            std::cout << path.string() << ": " << ranks.size() << " ranks, " << pairs / 2 << " neighbour pairs\n";
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
