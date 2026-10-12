// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Static vs dynamic set algebra: the price of the runtime version.
//
// The static algebra (include/samurai/subset/) encodes the whole expression
// in the C++ type system: the traversal is fully inlined, no virtual calls,
// no heap. The dynamic algebra (include/samurai/subset/dynamic/) trades that
// for a tree decided at runtime: each node is heap-allocated, an N-ary
// operator is a chain of binary ones, and every interval flows through a
// virtual call.
//
// Each pattern is measured in both forms, side by side, so the ratio is read
// directly off the report. Both build the expression inside the timed loop,
// as the library does (the mesh, hence the set, changes every step): the
// dynamic figure is the end-to-end cost, node construction included.
//
// Inputs (set_algebra_fixtures.hpp): a front at one level (thin shell) for
// the operators, and the MR-adapted meshes of benchmark_set_call_sites.cpp
// for two library expressions.

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "set_algebra_fixtures.hpp"

#include <samurai/subset/dynamic/dynamic.hpp>

namespace
{
    template <std::size_t dim>
    constexpr std::size_t front_level()
    {
        return (dim == 2) ? 11 : 8;
    }

    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& front()
    {
        static const auto lca = bench::make_shape<dim>(bench::Shape::shell, front_level<dim>());
        return lca;
    }

    enum class Algebra
    {
        static_,
        dynamic,
    };

    // intersection(self(a), translate(a, t[0]), ...) in either algebra.
    template <Algebra algebra, class LCA, class Translations, std::size_t... Is>
    auto make_intersection(const LCA& lca, const Translations& t, std::index_sequence<Is...> seq)
    {
        if constexpr (algebra == Algebra::static_)
        {
            return bench::make_nary(bench::intersection_op, lca, t, seq);
        }
        else
        {
            return samurai::dyn::intersection(samurai::dyn::self(lca), samurai::dyn::translate(samurai::dyn::self(lca), t[Is])...);
        }
    }

    // intersection(a, N - 1 unit translates of a), read at its own level or
    // at L - 1.
    template <Algebra algebra, std::size_t dim, std::size_t N, bool coarsen>
    void run_intersection(benchmark::State& state)
    {
        const auto& lca = front<dim>();
        const auto t    = bench::translations<dim, N - 1>(bench::Reach::unit);
        bench::run(state,
                   [&]
                   {
                       auto set = make_intersection<algebra>(lca, t, std::make_index_sequence<N - 1>{});
                       if constexpr (coarsen)
                       {
                           return set.on(front_level<dim>() - 1);
                       }
                       else
                       {
                           return set;
                       }
                   });
    }

    template <std::size_t dim>
    const auto& adapted_mesh(benchmark::State& state)
    {
        const auto& c = bench::adapted_case<dim>(static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(1)));
        bench::report_mesh(state, c.mesh);
        return c.mesh;
    }

    // The ghost projection of update_ghost_mr, every level.
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

    // The same-level faces of for_each_interface, every level and direction.
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

// --- Operators on a front ------------------------------------------------------

// intersection(a, N - 1 unit translates of a).
template <std::size_t dim, std::size_t N>
static void BM_Intersection_static(benchmark::State& state)
{
    run_intersection<Algebra::static_, dim, N, false>(state);
}

template <std::size_t dim, std::size_t N>
static void BM_Intersection_dynamic(benchmark::State& state)
{
    run_intersection<Algebra::dynamic, dim, N, false>(state);
}

// intersection(a, translate(a, e_x)).on(L - 1).
template <std::size_t dim>
static void BM_IntersectionCoarsen_static(benchmark::State& state)
{
    run_intersection<Algebra::static_, dim, 2, true>(state);
}

template <std::size_t dim>
static void BM_IntersectionCoarsen_dynamic(benchmark::State& state)
{
    run_intersection<Algebra::dynamic, dim, 2, true>(state);
}

BENCHMARK(BM_Intersection_static<2, 2>);
BENCHMARK(BM_Intersection_dynamic<2, 2>);
BENCHMARK(BM_Intersection_static<2, 3>);
BENCHMARK(BM_Intersection_dynamic<2, 3>);
BENCHMARK(BM_Intersection_static<2, 5>);
BENCHMARK(BM_Intersection_dynamic<2, 5>);
BENCHMARK(BM_Intersection_static<3, 2>);
BENCHMARK(BM_Intersection_dynamic<3, 2>);
BENCHMARK(BM_Intersection_static<3, 3>);
BENCHMARK(BM_Intersection_dynamic<3, 3>);
BENCHMARK(BM_Intersection_static<3, 5>);
BENCHMARK(BM_Intersection_dynamic<3, 5>);
BENCHMARK(BM_IntersectionCoarsen_static<2>);
BENCHMARK(BM_IntersectionCoarsen_dynamic<2>);
BENCHMARK(BM_IntersectionCoarsen_static<3>);
BENCHMARK(BM_IntersectionCoarsen_dynamic<3>);

// --- Library expressions on an adapted mesh --------------------------------------

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
