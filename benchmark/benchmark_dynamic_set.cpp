// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Price of the dynamic set algebra (include/samurai/subset/dynamic/) against
// the static one, on two expressions the library evaluates, on the MR-adapted
// meshes of benchmark_set_call_sites.cpp. Each benchmark builds the expression
// inside the timed loop, as the library does, so the dynamic figure includes
// the construction of its heap-allocated nodes.

#include <cstddef>

#include <benchmark/benchmark.h>

#include "set_algebra_fixtures.hpp"

#include <samurai/subset/dynamic/dynamic.hpp>

namespace
{
    enum class Algebra
    {
        static_,
        dynamic,
    };

    template <std::size_t dim>
    const auto& adapted_mesh(benchmark::State& state)
    {
        const auto& c = bench::adapted_case<dim>(static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(1)));
        bench::report_mesh(state, c.mesh);
        return c.mesh;
    }

    // update_ghost_mr (algorithm/update_ghost_mr.hpp): projection of the
    // ghosts, a coarsening.
    template <Algebra algebra, class Mesh, class F>
    void site_ghost_projection(const Mesh& mesh, F&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for (std::size_t level = mesh.max_level(); level >= 1; --level)
        {
            const auto& reference  = mesh[mesh_id_t::reference][level];
            const auto& proj_cells = mesh[mesh_id_t::proj_cells][level - 1];
            if constexpr (algebra == Algebra::static_)
            {
                f(level - 1, samurai::intersection(reference, proj_cells).on(level - 1));
            }
            else
            {
                f(level - 1, samurai::dyn::intersection(samurai::dyn::self(reference), samurai::dyn::self(proj_cells)).on(level - 1));
            }
        }
    }

    // for_each_interface (interface.hpp): faces between cells of the same
    // level, a same-level translate.
    template <Algebra algebra, class Mesh, class F>
    void site_interfaces(const Mesh& mesh, F&& f)
    {
        using mesh_id_t   = typename Mesh::mesh_id_t;
        const auto& cells = mesh[mesh_id_t::cells];
        for (std::size_t d = 0; d < Mesh::dim; ++d)
        {
            const auto direction = bench::unit<Mesh::dim>(d, -1);
            for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
            {
                if constexpr (algebra == Algebra::static_)
                {
                    f(level, samurai::intersection(cells[level], samurai::translate(cells[level], direction)));
                }
                else
                {
                    f(level,
                      samurai::dyn::intersection(samurai::dyn::self(cells[level]),
                                                 samurai::dyn::translate(samurai::dyn::self(cells[level]), direction)));
                }
            }
        }
    }

    template <Algebra algebra, std::size_t dim>
    void run_ghost_projection(benchmark::State& state)
    {
        const auto& mesh = adapted_mesh<dim>(state);
        bench::run_site(state,
                        [&](auto&& f)
                        {
                            site_ghost_projection<algebra>(mesh, f);
                        });
    }

    template <Algebra algebra, std::size_t dim>
    void run_interfaces(benchmark::State& state)
    {
        const auto& mesh = adapted_mesh<dim>(state);
        bench::run_site(state,
                        [&](auto&& f)
                        {
                            site_interfaces<algebra>(mesh, f);
                        });
    }
} // namespace

// Each _static / _dynamic pair evaluates one expression in both algebras:
// their ratio is the price of the runtime tree on what the library runs.
template <std::size_t dim>
static void BM_GhostProjection_static(benchmark::State& state)
{
    run_ghost_projection<Algebra::static_, dim>(state);
}

template <std::size_t dim>
static void BM_GhostProjection_dynamic(benchmark::State& state)
{
    run_ghost_projection<Algebra::dynamic, dim>(state);
}

template <std::size_t dim>
static void BM_Interfaces_static(benchmark::State& state)
{
    run_interfaces<Algebra::static_, dim>(state);
}

template <std::size_t dim>
static void BM_Interfaces_dynamic(benchmark::State& state)
{
    run_interfaces<Algebra::dynamic, dim>(state);
}

namespace
{
    template <std::size_t dim>
    void mesh_args(benchmark::internal::Benchmark* b)
    {
        b->Args({static_cast<long>(bench::default_max_level<dim>()), (dim == 2) ? 16 : 8});
        b->Unit(benchmark::kMicrosecond);
    }
} // namespace

BENCHMARK(BM_GhostProjection_static<2>)->Apply(mesh_args<2>);
BENCHMARK(BM_GhostProjection_dynamic<2>)->Apply(mesh_args<2>);
BENCHMARK(BM_GhostProjection_static<3>)->Apply(mesh_args<3>);
BENCHMARK(BM_GhostProjection_dynamic<3>)->Apply(mesh_args<3>);
BENCHMARK(BM_Interfaces_static<2>)->Apply(mesh_args<2>);
BENCHMARK(BM_Interfaces_dynamic<2>)->Apply(mesh_args<2>);
BENCHMARK(BM_Interfaces_static<3>)->Apply(mesh_args<3>);
BENCHMARK(BM_Interfaces_dynamic<3>)->Apply(mesh_args<3>);

BENCHMARK_MAIN();
