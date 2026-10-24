// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// make_graduation, the MR path, on the cells the adaptation passes to it: the
// cells of the next step before graduation (SiteInputs::ungraded_cells), read
// from the files of generate_set_algebra_inputs. make_graduation takes
// CellArrays only, so this binary needs no mesh object. It is a pipeline of
// materialized sets, so its whole time is set algebra and the building of
// LevelCellArrays.
//
// Args: (max_level, number of balls, stencil radius, geometry), see
// bench::mesh_cases.

#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>

#include <samurai/algorithm/graduation.hpp>

#include "site_inputs.hpp"

namespace
{
    // make_graduation takes the MPI neighbours of the mesh, none here.
    struct NoMesh
    {
    };
} // namespace

// Reports the cells before (cells_in) and after (cells_out) graduation, to
// show the closure has work to do.
template <std::size_t dim>
static void BM_Graduation(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    const std::vector<samurai::MPI_Subdomain<NoMesh>> no_neighbours;

    const auto graduate = [&](auto& ca)
    {
        samurai::make_graduation(ca, in.domain_pyramid, no_neighbours, in.periodic, in.graduation_width, in.max_stencil_radius);
    };

    for (auto _ : state)
    {
        state.PauseTiming();
        auto ca = in.ungraded_cells;
        state.ResumeTiming();
        graduate(ca);
        benchmark::DoNotOptimize(ca);
    }

    auto graded = in.ungraded_cells;
    graduate(graded);
    state.counters["cells_in"]  = static_cast<double>(in.ungraded_cells.nb_cells());
    state.counters["cells_out"] = static_cast<double>(graded.nb_cells());
}

BENCHMARK(BM_Graduation<1>)->Apply(bench::mesh_args<1>);
BENCHMARK(BM_Graduation<2>)->Apply(bench::mesh_args<2>);
BENCHMARK(BM_Graduation<3>)->Apply(bench::mesh_args<3>);

BENCHMARK_MAIN();
