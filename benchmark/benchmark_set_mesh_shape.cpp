// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// How the shape of a level drives the cost of the set algebra, in 1D, 2D and
// 3D. To start a row, a traverser searches the intervals of the parent row
// linearly (LCAView::get_traverser_impl), so the cost should grow with the
// length of the parent rows.
//
// Inputs: a box cut into K stripes along one axis (half of them kept, so the
// number of cells does not change with K), and a front cut into P slabs.

#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>

#include "set_algebra_fixtures.hpp"

namespace
{
    using bench::run;
    using bench::Shape;

    template <std::size_t dim>
    constexpr std::size_t shape_level()
    {
        return (dim == 1) ? 20 : ((dim == 2) ? 11 : 8);
    }

    template <std::size_t dim>
    samurai::LevelCellArray<dim> stripes(benchmark::State& state)
    {
        const auto axis = static_cast<std::size_t>(state.range(0));
        const auto k    = static_cast<std::size_t>(state.range(1));
        auto lca        = bench::make_shape<dim>(Shape::stripes, shape_level<dim>(), axis, k);
        bench::report_input(state, lca);
        return lca;
    }

    // Args: (axis, K), K up to slabs of one cell wide.
    template <std::size_t dim>
    void stripes_args(benchmark::internal::Benchmark* b)
    {
        const long kmax = long(1) << (shape_level<dim>() - 2);
        for (long axis = 0; axis < long(dim); ++axis)
        {
            for (long k = 1; k <= kmax; k *= 4)
            {
                b->Args({axis, k});
            }
        }
    }
} // namespace

// --- Fragmentation ----------------------------------------------------------------

// A plain traversal, a coarsening and an expansion of the same cells as the
// box is cut into more stripes along each axis. Along x the cut only adds
// output intervals; along the other axes it lengthens the parent rows that
// every row lookup scans. Args: (axis, K).
template <std::size_t dim>
static void BM_FragmentedSelf(benchmark::State& state)
{
    const auto lca = stripes<dim>(state);
    run(state,
        [&]
        {
            return samurai::self(lca);
        });
}

template <std::size_t dim>
static void BM_FragmentedCoarsen(benchmark::State& state)
{
    const auto lca = stripes<dim>(state);
    run(state,
        [&]
        {
            return samurai::self(lca).on(shape_level<dim>() - 1);
        });
}

template <std::size_t dim>
static void BM_FragmentedExpand(benchmark::State& state)
{
    const auto lca = stripes<dim>(state);
    run(state,
        [&]
        {
            return samurai::expand(lca, 1);
        });
}

#define SAMURAI_BENCH_FRAGMENTED(DIM)                               \
    BENCHMARK(BM_FragmentedSelf<DIM>)->Apply(stripes_args<DIM>);    \
    BENCHMARK(BM_FragmentedCoarsen<DIM>)->Apply(stripes_args<DIM>); \
    BENCHMARK(BM_FragmentedExpand<DIM>)->Apply(stripes_args<DIM>);

SAMURAI_BENCH_FRAGMENTED(1)
SAMURAI_BENCH_FRAGMENTED(2)
SAMURAI_BENCH_FRAGMENTED(3)

// --- Partitioning -------------------------------------------------------------------

// The same front seen by P processes, each intersecting it with its slab:
// does the total work grow with P, and does it depend on the axis of the cut?
// Sequential, so only the cost of the partition shape is measured. Args:
// (axis, P).
template <std::size_t dim>
static void BM_Partitioned(benchmark::State& state)
{
    using value_t       = typename samurai::LevelCellArray<dim>::value_t;
    const auto axis     = static_cast<std::size_t>(state.range(0));
    const auto parts    = static_cast<value_t>(state.range(1));
    const auto level    = shape_level<dim>();
    const value_t n     = value_t(1) << level;
    const auto front    = bench::make_front<dim>(level);
    const value_t width = n / parts;

    std::vector<samurai::LevelCellArray<dim>> subdomains;
    for (value_t p = 0; p < parts; ++p)
    {
        samurai::Box<value_t, dim> slab(xt::zeros<value_t>({dim}), xt::ones<value_t>({dim}) * n);
        slab.min_corner()[axis] = p * width;
        slab.max_corner()[axis] = (p + 1) * width;
        subdomains.emplace_back(level, slab);
    }

    bench::Tally tally;
    for (auto _ : state)
    {
        tally = {};
        for (const auto& subdomain : subdomains)
        {
            tally += bench::traverse(samurai::intersection(front, subdomain));
        }
    }
    bench::report(state, tally);
    bench::report_input(state, front);
}

template <std::size_t dim>
void partition_args(benchmark::internal::Benchmark* b)
{
    for (long axis = 0; axis < long(dim); ++axis)
    {
        for (long p : {1, 4, 16, 64})
        {
            b->Args({axis, p});
        }
    }
}

BENCHMARK(BM_Partitioned<1>)->Apply(partition_args<1>);
BENCHMARK(BM_Partitioned<2>)->Apply(partition_args<2>);
BENCHMARK(BM_Partitioned<3>)->Apply(partition_args<3>);

BENCHMARK_MAIN();
