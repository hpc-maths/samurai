// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// How the shape of a level, and the way it is cut, drives the cost of the
// set algebra, in 1D, 2D and 3D.
//
// A LevelCellArray stores, per dimension, the intervals of each row and the
// offsets that link a row to its sub-rows. To start a row, a traverser looks
// up the interval holding its coordinate among the intervals of the parent
// row (LCAView::get_traverser_impl). That lookup is linear, so its cost grows
// with the number of intervals of the parent row.
//
// Sections:
//
//   1. Fragmentation: the same number of cells (a box cut into K stripes, half
//      of them kept) as K grows, along each axis in turn. Along x (axis 0) K
//      only adds intervals per row; along the other axes it adds intervals to
//      the parent rows that every row lookup scans.
//   2. Partitioning: the same front traversed as P subdomains, the way P MPI
//      processes would see it (each process intersects the mesh with its
//      subdomain), for slabs cut along each axis.

#include <array>
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

    // Args: (axis, K) with K = 1, 4, ..., up to slabs of one cell.
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

// --- 1. Fragmentation ------------------------------------------------------------

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

// intersection(a, translate(a, e_x)): the flux / stencil pattern.
template <std::size_t dim>
static void BM_FragmentedTranslateX(benchmark::State& state)
{
    const auto lca = stripes<dim>(state);
    const auto t   = bench::unit<dim>(0);
    run(state,
        [&]
        {
            return samurai::intersection(lca, samurai::translate(lca, t));
        });
}

// intersection(a, translate(a, e_outer)): the same along the outermost axis,
// where the translate reads other rows than its partner.
template <std::size_t dim>
static void BM_FragmentedTranslateOuter(benchmark::State& state)
{
    const auto lca = stripes<dim>(state);
    const auto t   = bench::unit<dim>(dim - 1);
    run(state,
        [&]
        {
            return samurai::intersection(lca, samurai::translate(lca, t));
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

#define SAMURAI_BENCH_FRAGMENTED(DIM)                                      \
    BENCHMARK(BM_FragmentedSelf<DIM>)->Apply(stripes_args<DIM>);           \
    BENCHMARK(BM_FragmentedTranslateX<DIM>)->Apply(stripes_args<DIM>);     \
    BENCHMARK(BM_FragmentedTranslateOuter<DIM>)->Apply(stripes_args<DIM>); \
    BENCHMARK(BM_FragmentedCoarsen<DIM>)->Apply(stripes_args<DIM>);        \
    BENCHMARK(BM_FragmentedExpand<DIM>)->Apply(stripes_args<DIM>);

SAMURAI_BENCH_FRAGMENTED(1)
SAMURAI_BENCH_FRAGMENTED(2)
SAMURAI_BENCH_FRAGMENTED(3)

// --- 2. Partitioning ----------------------------------------------------------------

// The front (a shell; many segments in 1D) intersected with each of P slabs
// cut along `axis`, all of them traversed: the total work of P processes.
// Args: (axis, P).
template <std::size_t dim>
static void BM_Partitioned(benchmark::State& state)
{
    using value_t       = typename samurai::LevelCellArray<dim>::value_t;
    const auto axis     = static_cast<std::size_t>(state.range(0));
    const auto parts    = static_cast<value_t>(state.range(1));
    const auto level    = shape_level<dim>();
    const value_t n     = value_t(1) << level;
    const auto front    = (dim == 1) ? bench::make_shape<dim>(Shape::balls, level, 0, 4096) : bench::make_shape<dim>(Shape::shell, level);
    const value_t width = n / parts;

    std::vector<samurai::LevelCellArray<dim>> subdomains;
    for (value_t p = 0; p < parts; ++p)
    {
        std::array<value_t, dim> lo{};
        std::array<value_t, dim> hi;
        hi.fill(n);
        lo[axis] = p * width;
        hi[axis] = (p + 1) * width;
        subdomains.push_back(bench::make_box<dim>(level, lo, hi));
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
