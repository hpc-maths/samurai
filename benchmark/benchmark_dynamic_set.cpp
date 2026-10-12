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

    template <std::size_t dim, std::size_t N>
    std::array<xt::xtensor_fixed<int, xt::xshape<dim>>, N> translations()
    {
        std::array<xt::xtensor_fixed<int, xt::xshape<dim>>, N> t;
        for (std::size_t i = 0; i < N; ++i)
        {
            t[i] = bench::unit<dim>((i / 2) % dim, (i % 2 == 0) ? 1 : -1);
        }
        return t;
    }

    template <class LCA, class Translations, std::size_t... Is>
    auto static_intersection(const LCA& lca, const Translations& t, std::index_sequence<Is...>)
    {
        return samurai::intersection(samurai::self(lca), samurai::translate(lca, t[Is])...);
    }

    template <class LCA, class Translations, std::size_t... Is>
    auto dynamic_intersection(const LCA& lca, const Translations& t, std::index_sequence<Is...>)
    {
        return samurai::dyn::intersection(samurai::dyn::self(lca), samurai::dyn::translate(samurai::dyn::self(lca), t[Is])...);
    }

    template <class Make>
    void run(benchmark::State& state, Make&& make)
    {
        bench::Tally tally;
        for (auto _ : state)
        {
            tally = bench::traverse(make());
        }
        bench::report(state, tally);
    }

    template <class Site>
    void run_site(benchmark::State& state, Site&& site)
    {
        bench::Tally tally;
        for (auto _ : state)
        {
            tally = {};
            site(
                [&](const auto& set)
                {
                    tally += bench::traverse(set);
                });
        }
        bench::report(state, tally);
    }

    template <std::size_t dim>
    auto& adapted_mesh(benchmark::State& state)
    {
        auto& c = bench::adapted_case<dim>(static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(1)));
        bench::report_mesh(state, c.mesh);
        return c.mesh;
    }
} // namespace

// --- Operators on a front ------------------------------------------------------

// intersection(a, N - 1 unit translates of a).
template <std::size_t dim, std::size_t N>
static void BM_Intersection_static(benchmark::State& state)
{
    const auto& lca = front<dim>();
    const auto t    = translations<dim, N - 1>();
    run(state,
        [&]
        {
            return static_intersection(lca, t, std::make_index_sequence<N - 1>{});
        });
}

template <std::size_t dim, std::size_t N>
static void BM_Intersection_dynamic(benchmark::State& state)
{
    const auto& lca = front<dim>();
    const auto t    = translations<dim, N - 1>();
    run(state,
        [&]
        {
            return dynamic_intersection(lca, t, std::make_index_sequence<N - 1>{});
        });
}

// intersection(a, translate(a, e_x)).on(L - 1).
template <std::size_t dim>
static void BM_IntersectionCoarsen_static(benchmark::State& state)
{
    const auto& lca = front<dim>();
    const auto t    = translations<dim, 1>();
    run(state,
        [&]
        {
            return static_intersection(lca, t, std::make_index_sequence<1>{}).on(front_level<dim>() - 1);
        });
}

template <std::size_t dim>
static void BM_IntersectionCoarsen_dynamic(benchmark::State& state)
{
    const auto& lca = front<dim>();
    const auto t    = translations<dim, 1>();
    run(state,
        [&]
        {
            return dynamic_intersection(lca, t, std::make_index_sequence<1>{}).on(front_level<dim>() - 1);
        });
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

// The ghost projection of update_ghost_mr, every level.
template <std::size_t dim>
static void BM_GhostProjection_static(benchmark::State& state)
{
    const auto& mesh = adapted_mesh<dim>(state);
    using mesh_id_t  = typename std::decay_t<decltype(mesh)>::mesh_id_t;
    run_site(state,
             [&](auto&& f)
             {
                 for (std::size_t level = mesh.max_level(); level >= 1; --level)
                 {
                     f(samurai::intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1));
                 }
             });
}

template <std::size_t dim>
static void BM_GhostProjection_dynamic(benchmark::State& state)
{
    const auto& mesh = adapted_mesh<dim>(state);
    using mesh_id_t  = typename std::decay_t<decltype(mesh)>::mesh_id_t;
    run_site(state,
             [&](auto&& f)
             {
                 for (std::size_t level = mesh.max_level(); level >= 1; --level)
                 {
                     f(samurai::dyn::intersection(samurai::dyn::self(mesh[mesh_id_t::reference][level]),
                                                  samurai::dyn::self(mesh[mesh_id_t::proj_cells][level - 1]))
                           .on(level - 1));
                 }
             });
}

// The same-level faces of for_each_interface, every level and direction.
template <std::size_t dim>
static void BM_Interfaces_static(benchmark::State& state)
{
    const auto& mesh  = adapted_mesh<dim>(state);
    using mesh_id_t   = typename std::decay_t<decltype(mesh)>::mesh_id_t;
    const auto& cells = mesh[mesh_id_t::cells];
    run_site(state,
             [&](auto&& f)
             {
                 for (std::size_t d = 0; d < dim; ++d)
                 {
                     const auto direction = bench::unit<dim>(d);
                     for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
                     {
                         f(samurai::intersection(cells[level], samurai::translate(cells[level], -direction)));
                     }
                 }
             });
}

template <std::size_t dim>
static void BM_Interfaces_dynamic(benchmark::State& state)
{
    const auto& mesh  = adapted_mesh<dim>(state);
    using mesh_id_t   = typename std::decay_t<decltype(mesh)>::mesh_id_t;
    const auto& cells = mesh[mesh_id_t::cells];
    run_site(state,
             [&](auto&& f)
             {
                 for (std::size_t d = 0; d < dim; ++d)
                 {
                     const xt::xtensor_fixed<int, xt::xshape<dim>> direction = -bench::unit<dim>(d);
                     for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
                     {
                         f(samurai::dyn::intersection(samurai::dyn::self(cells[level]),
                                                      samurai::dyn::translate(samurai::dyn::self(cells[level]), direction)));
                     }
                 }
             });
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
