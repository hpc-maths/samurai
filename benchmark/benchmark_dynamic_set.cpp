// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Price of the dynamic set algebra (include/samurai/subset/dynamic/) against
// the static one, on two expressions the library evaluates, on the sets of the
// MR-adapted meshes of bench_set_call_sites (read from the files of
// generate_set_algebra_inputs). Each benchmark builds the expression inside
// the timed loop, as the library does, so the dynamic figure includes the
// construction of its heap-allocated nodes.

#include <cstddef>

#include <benchmark/benchmark.h>

#include <xtensor/views/xview.hpp>

#include <samurai/stencil.hpp>
#include <samurai/subset/dynamic/dynamic.hpp>
#include <samurai/subset/node.hpp>

#include "set_algebra_fixtures.hpp"
#include "site_inputs.hpp"

namespace
{
    enum class Algebra
    {
        static_,
        dynamic,
    };

    // update_ghost_mr (algorithm/update_ghost_mr.hpp): projection of the
    // ghosts, a coarsening.
    template <Algebra algebra, std::size_t dim, class F>
    void site_ghost_projection(const bench::SiteInputs<dim>& in, F&& f)
    {
        for (std::size_t level = in.max_level; level >= 1; --level)
        {
            const auto& reference  = in.reference[level];
            const auto& proj_cells = in.proj_cells[level - 1];
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
    template <Algebra algebra, std::size_t dim, class F>
    void site_interfaces(const bench::SiteInputs<dim>& in, F&& f)
    {
        const auto& cells     = in.cells;
        const auto directions = samurai::positive_cartesian_directions<dim>();
        for (std::size_t d = 0; d < dim; ++d)
        {
            const samurai::DirectionVector<dim> direction = -xt::view(directions, d);
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
        const auto& in = bench::site_inputs<dim>(state);
        bench::run_site(state,
                        [&](auto&& f)
                        {
                            site_ghost_projection<algebra>(in, f);
                        });
    }

    template <Algebra algebra, std::size_t dim>
    void run_interfaces(benchmark::State& state)
    {
        const auto& in = bench::site_inputs<dim>(state);
        bench::run_site(state,
                        [&](auto&& f)
                        {
                            site_interfaces<algebra>(in, f);
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
    // One of bench::mesh_cases: the most fronts at the default max level.
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
