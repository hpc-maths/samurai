// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Time the set algebra takes in the operations samurai runs at every time
// step, on the sets of MR-adapted meshes in 1D, 2D and 3D. Needs only the
// interval structures and the set algebra: the sets are read from the files
// of generate_set_algebra_inputs. bench_set_library_ops times the same
// operations with the library machinery.
//
// Args: (max_level, number of balls), see bench::mesh_cases.

#include <cstddef>
#include <utility>

#include <benchmark/benchmark.h>

#include "set_algebra_fixtures.hpp"
#include "set_call_sites.hpp"
#include "site_inputs.hpp"

namespace
{
    template <class Site>
    void run_site(benchmark::State& state, Site&& site)
    {
        state.counters["sets"] = static_cast<double>(bench::run_site(state, std::forward<Site>(site)));
    }
} // namespace

namespace sites = bench::sites;

// BM_Site_X traverses the sets of site X with a callback that only counts:
// the time the set algebra takes in the library function, to set against
// BM_SiteOp_X or BM_Op_* of bench_set_library_ops.
template <std::size_t dim>
static void BM_Site_MeshPredictionGhosts(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::mesh_prediction_ghosts(in, sites::projected_subdomain(in), f);
             });
}

template <std::size_t dim>
static void BM_Site_Corners(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::corners(in, f);
             });
}

template <std::size_t dim>
static void BM_Site_GhostProjection(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::ghost_projection(in, f);
             });
}

template <std::size_t dim>
static void BM_Site_GhostPrediction(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::ghost_prediction(in, f);
             });
}

template <std::size_t dim>
static void BM_Site_OuterGhosts(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::outer_ghosts(in, sites::projected_domain(in), f);
             });
}

// BM_SiteCached_X: site X reading the coarse domain or subdomain the mesh
// already stores instead of coarsening the finest one at every use. The gap
// with BM_Site_X is what caching coarse sets would save.
template <std::size_t dim>
static void BM_SiteCached_OuterGhosts(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::outer_ghosts(in, sites::cached_domain(in), f);
             });
}

template <std::size_t dim>
static void BM_SiteCached_MeshPredictionGhosts(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::mesh_prediction_ghosts(in, sites::cached_subdomain(in), f);
             });
}

template <std::size_t dim>
static void BM_Site_Detail(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::detail(in, f);
             });
}

template <std::size_t dim>
static void BM_Site_Tag(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::tag(in, f);
             });
}

template <std::size_t dim>
static void BM_Site_Interfaces(benchmark::State& state)
{
    const auto& in = bench::site_inputs<dim>(state);
    run_site(state,
             [&](auto&& f)
             {
                 sites::interfaces(in, f);
             });
}

template <std::size_t dim>
static void BM_Site_FieldTransfer(benchmark::State& state)
{
    const auto& in     = bench::site_inputs<dim>(state);
    const auto& new_in = bench::site_inputs<dim>(state, true);
    run_site(state,
             [&](auto&& f)
             {
                 sites::field_transfer(in, new_in, f);
             });
}

#define SAMURAI_BENCH_CALL_SITES(DIM)                                                 \
    BENCHMARK(BM_Site_MeshPredictionGhosts<DIM>)->Apply(bench::mesh_args<DIM>);       \
    BENCHMARK(BM_SiteCached_MeshPredictionGhosts<DIM>)->Apply(bench::mesh_args<DIM>); \
    BENCHMARK(BM_Site_Corners<DIM>)->Apply(bench::mesh_args<DIM>);                    \
    BENCHMARK(BM_Site_GhostProjection<DIM>)->Apply(bench::mesh_args<DIM>);            \
    BENCHMARK(BM_Site_GhostPrediction<DIM>)->Apply(bench::mesh_args<DIM>);            \
    BENCHMARK(BM_Site_OuterGhosts<DIM>)->Apply(bench::mesh_args<DIM>);                \
    BENCHMARK(BM_SiteCached_OuterGhosts<DIM>)->Apply(bench::mesh_args<DIM>);          \
    BENCHMARK(BM_Site_Detail<DIM>)->Apply(bench::mesh_args<DIM>);                     \
    BENCHMARK(BM_Site_Tag<DIM>)->Apply(bench::mesh_args<DIM>);                        \
    BENCHMARK(BM_Site_FieldTransfer<DIM>)->Apply(bench::mesh_args<DIM>);              \
    BENCHMARK(BM_Site_Interfaces<DIM>)->Apply(bench::mesh_args<DIM>);

SAMURAI_BENCH_CALL_SITES(1)
SAMURAI_BENCH_CALL_SITES(2)
SAMURAI_BENCH_CALL_SITES(3)

BENCHMARK_MAIN();
