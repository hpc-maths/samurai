// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Cost of moving a set between levels, in 1D, 2D and 3D.
//
// A set at level L is read at another level either explicitly, with .on(l),
// or implicitly, when the operands of an intersection / union / difference
// live at different levels (the operator works at the finest one and refines
// the others). The coarsening .on(l < L) builds, for every coarse row, the
// list of the intervals of the 2^(L - l) fine rows it covers per dimension
// (subset_utils::transform_to_loi): that temporary list is the suspected
// bottleneck when going from a fine to a coarse level.
//
// Sections:
//
//   1. Coarsening against the level difference s, compared with traversing
//      the same set already built at the coarse level (the floor), and with
//      the cost of building it once.
//   2. Refinement against s.
//   3. Coarsening of a composite expression: an N-ary intersection read one
//      level below, against the same intersection at its own level.
//   4. Chained projections, as written in the graduation (.on(l-1).on(l)).
//   5. Expansion of a coarsened set, the mesh construction pattern
//      (expand(self(cells[l]).on(l-2), r)).
//   6. Implicit level change: an intersection between operands at different
//      levels, with and without .on() to the coarse level.

#include <array>
#include <cstddef>
#include <utility>

#include <benchmark/benchmark.h>

#include "set_algebra_fixtures.hpp"

namespace
{
    using bench::Shape;

    template <std::size_t dim>
    constexpr std::size_t fine_level()
    {
        return (dim == 1) ? 20 : ((dim == 2) ? 11 : 8);
    }

    // The fine set: a front (thin shell) in 2D and 3D, many segments in 1D.
    template <std::size_t dim>
    samurai::LevelCellArray<dim> make_front(std::size_t level)
    {
        return (dim == 1) ? bench::make_shape<dim>(Shape::balls, level, 0, 4096) : bench::make_shape<dim>(Shape::shell, level);
    }

    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& fine(Shape shape)
    {
        static const auto front = make_front<dim>(fine_level<dim>());
        static const auto box   = bench::make_shape<dim>(Shape::box, fine_level<dim>());
        return (shape == Shape::box) ? box : front;
    }

    template <class Make>
    bench::Tally run(benchmark::State& state, Make&& make)
    {
        bench::Tally tally;
        for (auto _ : state)
        {
            tally = bench::traverse(make());
        }
        bench::report(state, tally);
        return tally;
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
    auto make_intersection(const LCA& lca, const Translations& t, std::index_sequence<Is...>)
    {
        return samurai::intersection(samurai::self(lca), samurai::translate(lca, t[Is])...);
    }

    // Time per fine input row: the unit of work of the coarsening.
    template <std::size_t dim>
    void report_fine_rows(benchmark::State& state, const samurai::LevelCellArray<dim>& lca)
    {
        bench::report_input(state, lca);
        state.counters["t/in_row"] = benchmark::Counter(static_cast<double>(bench::count_rows(lca)),
                                                        benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    }
} // namespace

// --- 1. Coarsening against the level difference ------------------------------

// self(fine).on(L - s). Arg: s.
template <std::size_t dim, Shape shape>
static void BM_Coarsen(benchmark::State& state)
{
    const auto& lca     = fine<dim>(shape);
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    run(state,
        [&]
        {
            return samurai::self(lca).on(fine_level<dim>() - s);
        });
    report_fine_rows(state, lca);
}

// The floor of BM_Coarsen: the same coarse set, already built. Arg: s.
template <std::size_t dim, Shape shape>
static void BM_CoarsenPrebuilt(benchmark::State& state)
{
    const auto& lca     = fine<dim>(shape);
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    const samurai::LevelCellArray<dim> coarse(samurai::self(lca).on(fine_level<dim>() - s));
    run(state,
        [&]
        {
            return samurai::self(coarse);
        });
    report_fine_rows(state, lca);
}

// Building the coarse LevelCellArray once: what a cached coarse set would
// cost per mesh change. Arg: s.
template <std::size_t dim, Shape shape>
static void BM_CoarsenBuild(benchmark::State& state)
{
    const auto& lca     = fine<dim>(shape);
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    for (auto _ : state)
    {
        samurai::LevelCellArray<dim> coarse(samurai::self(lca).on(fine_level<dim>() - s));
        benchmark::DoNotOptimize(coarse);
    }
    report_fine_rows(state, lca);
}

#define SAMURAI_BENCH_COARSEN(DIM, SHAPE)                                              \
    BENCHMARK(BM_Coarsen<DIM, SHAPE>)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(6);         \
    BENCHMARK(BM_CoarsenPrebuilt<DIM, SHAPE>)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(6); \
    BENCHMARK(BM_CoarsenBuild<DIM, SHAPE>)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(6);

SAMURAI_BENCH_COARSEN(1, Shape::shell)
SAMURAI_BENCH_COARSEN(2, Shape::shell)
SAMURAI_BENCH_COARSEN(2, Shape::box)
SAMURAI_BENCH_COARSEN(3, Shape::shell)
SAMURAI_BENCH_COARSEN(3, Shape::box)

// --- 2. Refinement against the level difference ------------------------------

// self(coarse).on(Lc + s), the coarse front at Lc = L - s: the output is the
// same region at the same resolution for every s. Arg: s.
//
// BM_Refine<3> is also a codegen canary: the fully inlined 3D refinement is so
// register-tight that perturbing the Projection code was measured to cost
// about +30% executed instructions (spills) on this path even when the
// perturbation is never executed (a std::vector member added to
// SetTraits<Projection>::Workspace, or a write to a Workspace member from the
// COARSEN branch of get_traverser_impl); noinline / cold / preserve_most on
// the added code did not help. Re-run it after any change to
// include/samurai/subset/, however innocuous.
template <std::size_t dim>
static void BM_Refine(benchmark::State& state)
{
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    const auto coarse   = make_front<dim>(fine_level<dim>() - s);
    run(state,
        [&]
        {
            return samurai::self(coarse).on(fine_level<dim>());
        });
    bench::report_input(state, coarse);
}

// The floor of BM_Refine: the refined set, already built. Arg: s.
template <std::size_t dim>
static void BM_RefinePrebuilt(benchmark::State& state)
{
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    const auto coarse   = make_front<dim>(fine_level<dim>() - s);
    const samurai::LevelCellArray<dim> refined(samurai::self(coarse).on(fine_level<dim>()));
    run(state,
        [&]
        {
            return samurai::self(refined);
        });
    bench::report_input(state, coarse);
}

BENCHMARK(BM_Refine<1>)->Arg(1)->Arg(2)->Arg(4)->Arg(6);
BENCHMARK(BM_RefinePrebuilt<1>)->Arg(1)->Arg(2)->Arg(4)->Arg(6);
BENCHMARK(BM_Refine<2>)->Arg(1)->Arg(2)->Arg(4)->Arg(6);
BENCHMARK(BM_RefinePrebuilt<2>)->Arg(1)->Arg(2)->Arg(4)->Arg(6);
BENCHMARK(BM_Refine<3>)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK(BM_RefinePrebuilt<3>)->Arg(1)->Arg(2)->Arg(4);

// --- 3. Coarsening of a composite expression ---------------------------------

// intersection(a, N - 1 unit translates of a), at its own level.
template <std::size_t dim, std::size_t N>
static void BM_Composite(benchmark::State& state)
{
    const auto& lca = fine<dim>(Shape::shell);
    const auto t    = translations<dim, N - 1>();
    run(state,
        [&]
        {
            return make_intersection(lca, t, std::make_index_sequence<N - 1>{});
        });
    report_fine_rows(state, lca);
}

// The same intersection read one level below: each coarse row re-evaluates
// the whole expression on the fine rows it covers.
template <std::size_t dim, std::size_t N>
static void BM_CompositeCoarsen(benchmark::State& state)
{
    const auto& lca = fine<dim>(Shape::shell);
    const auto t    = translations<dim, N - 1>();
    run(state,
        [&]
        {
            return make_intersection(lca, t, std::make_index_sequence<N - 1>{}).on(fine_level<dim>() - 1);
        });
    report_fine_rows(state, lca);
}

#define SAMURAI_BENCH_COMPOSITE(DIM)        \
    BENCHMARK(BM_Composite<DIM, 2>);        \
    BENCHMARK(BM_CompositeCoarsen<DIM, 2>); \
    BENCHMARK(BM_Composite<DIM, 3>);        \
    BENCHMARK(BM_CompositeCoarsen<DIM, 3>); \
    BENCHMARK(BM_Composite<DIM, 5>);        \
    BENCHMARK(BM_CompositeCoarsen<DIM, 5>);

SAMURAI_BENCH_COMPOSITE(1)
SAMURAI_BENCH_COMPOSITE(2)
SAMURAI_BENCH_COMPOSITE(3)

// --- 4. Chained projections ----------------------------------------------------

// self(a).on(L - 1).on(L): the rounding of a set up to whole coarse cells,
// as the graduation writes it.
template <std::size_t dim>
static void BM_CoarsenThenRefine(benchmark::State& state)
{
    const auto& lca = fine<dim>(Shape::shell);
    run(state,
        [&]
        {
            return samurai::self(lca).on(fine_level<dim>() - 1).on(fine_level<dim>());
        });
    report_fine_rows(state, lca);
}

// self(a).on(L - 1).on(L - 2) against BM_Coarsen/2, which reaches the same
// set in one projection.
template <std::size_t dim>
static void BM_CoarsenTwice(benchmark::State& state)
{
    const auto& lca = fine<dim>(Shape::shell);
    run(state,
        [&]
        {
            return samurai::self(lca).on(fine_level<dim>() - 1).on(fine_level<dim>() - 2);
        });
    report_fine_rows(state, lca);
}

BENCHMARK(BM_CoarsenThenRefine<1>);
BENCHMARK(BM_CoarsenThenRefine<2>);
BENCHMARK(BM_CoarsenThenRefine<3>);
BENCHMARK(BM_CoarsenTwice<1>);
BENCHMARK(BM_CoarsenTwice<2>);
BENCHMARK(BM_CoarsenTwice<3>);

// --- 5. Expansion of a coarsened set --------------------------------------------

// expand(self(a).on(L - s), 1): the prediction ghosts of the mesh
// construction (s = 1 and 2). Arg: s.
template <std::size_t dim>
static void BM_ExpandCoarsen(benchmark::State& state)
{
    const auto& lca     = fine<dim>(Shape::shell);
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    run(state,
        [&]
        {
            return samurai::expand(samurai::self(lca).on(fine_level<dim>() - s), 1);
        });
    report_fine_rows(state, lca);
}

// The same on a coarse set already built. Arg: s.
template <std::size_t dim>
static void BM_ExpandCoarsenPrebuilt(benchmark::State& state)
{
    const auto& lca     = fine<dim>(Shape::shell);
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    const samurai::LevelCellArray<dim> coarse(samurai::self(lca).on(fine_level<dim>() - s));
    run(state,
        [&]
        {
            return samurai::expand(coarse, 1);
        });
    report_fine_rows(state, lca);
}

BENCHMARK(BM_ExpandCoarsen<1>)->Arg(1)->Arg(2);
BENCHMARK(BM_ExpandCoarsenPrebuilt<1>)->Arg(1)->Arg(2);
BENCHMARK(BM_ExpandCoarsen<2>)->Arg(1)->Arg(2);
BENCHMARK(BM_ExpandCoarsenPrebuilt<2>)->Arg(1)->Arg(2);
BENCHMARK(BM_ExpandCoarsen<3>)->Arg(1)->Arg(2);
BENCHMARK(BM_ExpandCoarsenPrebuilt<3>)->Arg(1)->Arg(2);

// --- 6. Implicit level change ---------------------------------------------------

// intersection(fine front at L, coarse box at L - s): the operator works at L
// and refines the box on the fly. Arg: s.
template <std::size_t dim>
static void BM_MixedLevels(benchmark::State& state)
{
    const auto& lca     = fine<dim>(Shape::shell);
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    const auto coarse   = bench::make_shape<dim>(Shape::box, fine_level<dim>() - s);
    run(state,
        [&]
        {
            return samurai::intersection(lca, coarse);
        });
    report_fine_rows(state, lca);
}

// The same intersection read at the coarse level, the pattern of the
// projection and of the tagging: intersection(cells[l], all_cells[l-1]).on(l-1).
// Arg: s.
template <std::size_t dim>
static void BM_MixedLevelsCoarsen(benchmark::State& state)
{
    const auto& lca     = fine<dim>(Shape::shell);
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    const auto coarse   = bench::make_shape<dim>(Shape::box, fine_level<dim>() - s);
    run(state,
        [&]
        {
            return samurai::intersection(lca, coarse).on(fine_level<dim>() - s);
        });
    report_fine_rows(state, lca);
}

BENCHMARK(BM_MixedLevels<1>)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK(BM_MixedLevelsCoarsen<1>)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK(BM_MixedLevels<2>)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK(BM_MixedLevelsCoarsen<2>)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK(BM_MixedLevels<3>)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK(BM_MixedLevelsCoarsen<3>)->Arg(1)->Arg(2)->Arg(4);

BENCHMARK_MAIN();
