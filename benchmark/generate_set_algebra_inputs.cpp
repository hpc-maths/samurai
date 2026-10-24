// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Adapts the meshes of bench::mesh_cases, unmoved and moved, and writes their
// sets, and the cells of the next step before graduation with the sets
// update_cell_array_from_tag builds them from, to the files
// bench_set_call_sites, bench_dynamic_set and bench_set_graduation read
// (format in site_inputs.hpp). Then writes the partitions of
// bench::partition_cases, the meshes of their ranks. Each file is read back
// and compared with what it was written from. Next to each mesh and
// partition, writes the checksum of what each call site computes on it with
// the static algebra (site_checksum.hpp): the reference the benchmarks check
// every backend against.

#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "call_site_benchmarks.hpp"
#include "mr_fixtures.hpp"
#include "set_algebra_backend.hpp"
#include "site_checksum.hpp"

namespace
{
    // Runs each site of `for_each_site` on the context make_context() gives,
    // and writes their checksums to the checksum file of c.
    template <std::size_t dim, class ForEachSite, class MakeContext>
    void write_reference_checksums(const bench::MeshCase& c, ForEachSite&& for_each_site, MakeContext&& make_context)
    {
        std::map<std::string, bench::SiteChecksum> checksums;
        for_each_site(
            [&](const char* site, bool periodic_only, auto run)
            {
                if (periodic_only && c.geometry != bench::Geometry::periodic)
                {
                    return;
                }
                auto ctx = make_context();
                run(ctx);
                checksums[site] = ctx.checksum();
            });
        const auto path = bench::checksum_file<dim>(c);
        bench::write_checksums<dim>(checksums, path);
        if (bench::read_checksums<dim>(path) != checksums)
        {
            throw std::runtime_error(path.string() + " does not read back as the checksums it was written from");
        }
        std::cout << path.string() << ": " << checksums.size() << " sites\n";
    }

    template <std::size_t dim>
    void generate()
    {
        using static_algebra = bench::SamuraiStatic;
        for (const auto& c : bench::mesh_cases<dim>())
        {
            std::array<bench::SiteInputs<dim>, 2> inputs;
            for (const bool moved : {false, true})
            {
                const auto& mesh = bench::adapted_case<dim>(c, moved).mesh;
                auto& in         = inputs[moved ? 1 : 0];
                in               = bench::make_site_inputs<dim>(mesh);
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
            write_reference_checksums<dim>(
                c,
                [](auto&& v)
                {
                    bench::for_each_call_site<static_algebra, dim>(v);
                },
                [&]
                {
                    return bench::Recording<static_algebra, dim>(inputs[0], inputs[1]);
                });
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
            write_reference_checksums<dim>(
                c,
                [](auto&& v)
                {
                    bench::for_each_partition_site<static_algebra, dim>(v);
                },
                [&]
                {
                    return bench::Recording<static_algebra, dim>(ranks);
                });
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
