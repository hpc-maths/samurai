// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Cost of reading a set at another level with .on(l), in 1D, 2D and 3D. The
// coarsening is the suspect: for every coarse row it gathers the intervals of
// all the fine rows it covers (subset_utils::transform_to_loi), and nothing
// keeps the result between two evaluations.
//
// Input: a front at the fine level L (a thin shell in 2D and 3D, many
// segments in 1D). `t/in_row` is the time per fine row, the unit of work of
// the coarsening.

#include <array>
#include <cstddef>
#include <utility>

#include <benchmark/benchmark.h>

#include "set_algebra_fixtures.hpp"

namespace
{
    using bench::make_front;
    using bench::run;

    template <std::size_t dim>
    constexpr std::size_t fine_level()
    {
        return (dim == 1) ? 20 : ((dim == 2) ? 11 : 8);
    }

    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& fine()
    {
        static const auto front = make_front<dim>(fine_level<dim>());
        return front;
    }

    template <std::size_t dim>
    samurai::LevelCellArray<dim> coarsened(const samurai::LevelCellArray<dim>& lca, std::size_t s)
    {
        return samurai::LevelCellArray<dim>(samurai::self(lca).on(fine_level<dim>() - s));
    }

    template <std::size_t dim>
    void report_fine_rows(benchmark::State& state, const samurai::LevelCellArray<dim>& lca)
    {
        bench::report_input(state, lca);
        state.counters["t/in_row"] = benchmark::Counter(static_cast<double>(bench::count_rows(lca)),
                                                        benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    }
} // namespace

// --- Coarsening against the level difference --------------------------------

// The cost of self(a).on(L - s) per fine row as s grows, compared with
// traversing the coarse set already built (BM_CoarsenPrebuilt) and with
// building it once (BM_CoarsenBuild): the gain a cached coarse set would
// bring. Arg: s.
template <std::size_t dim>
static void BM_Coarsen(benchmark::State& state)
{
    const auto& lca     = fine<dim>();
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    run(state,
        [&]
        {
            return samurai::self(lca).on(fine_level<dim>() - s);
        });
    report_fine_rows(state, lca);
}

template <std::size_t dim>
static void BM_CoarsenPrebuilt(benchmark::State& state)
{
    const auto& lca   = fine<dim>();
    const auto coarse = coarsened(lca, static_cast<std::size_t>(state.range(0)));
    run(state,
        [&]
        {
            return samurai::self(coarse);
        });
    report_fine_rows(state, lca);
}

template <std::size_t dim>
static void BM_CoarsenBuild(benchmark::State& state)
{
    const auto& lca     = fine<dim>();
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    for (auto _ : state)
    {
        auto coarse = coarsened(lca, s);
        benchmark::DoNotOptimize(coarse);
    }
    report_fine_rows(state, lca);
}

#define SAMURAI_BENCH_COARSEN(DIM)                                              \
    BENCHMARK(BM_Coarsen<DIM>)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(6);         \
    BENCHMARK(BM_CoarsenPrebuilt<DIM>)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(6); \
    BENCHMARK(BM_CoarsenBuild<DIM>)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(6);

SAMURAI_BENCH_COARSEN(1)
SAMURAI_BENCH_COARSEN(2)
SAMURAI_BENCH_COARSEN(3)

// --- Refinement against the level difference --------------------------------

// Control for the coarsening: self(coarse).on(L) against the refined set
// already built (BM_RefinePrebuilt). The coarse front lives at L - s, so the
// output covers the same region at the same resolution for every s. Arg: s.
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

// --- Coarsening of a composite expression -----------------------------------

namespace
{
    template <std::size_t dim, std::size_t N, bool coarsen>
    void run_composite(benchmark::State& state)
    {
        const auto& lca = fine<dim>();
        const auto t    = bench::translations<dim, N - 1>(bench::Reach::unit);
        run(state,
            [&]
            {
                auto set = bench::make_nary(bench::intersection_op, lca, t, std::make_index_sequence<N - 1>{});
                if constexpr (coarsen)
                {
                    return set.on(fine_level<dim>() - 1);
                }
                else
                {
                    return set;
                }
            });
        report_fine_rows(state, lca);
    }
} // namespace

// An N-ary intersection read one level below (BM_CompositeCoarsen) against the
// same intersection at its own level (BM_Composite): does each coarse row
// re-evaluate the whole expression on the fine rows it covers?
template <std::size_t dim, std::size_t N>
static void BM_Composite(benchmark::State& state)
{
    run_composite<dim, N, false>(state);
}

template <std::size_t dim, std::size_t N>
static void BM_CompositeCoarsen(benchmark::State& state)
{
    run_composite<dim, N, true>(state);
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

// --- Chained projections --------------------------------------------------------

// self(a).on(L - 1).on(L), the rounding up to whole coarse cells that
// make_graduation (algorithm/graduation.hpp) writes: compare with BM_Coarsen/1
// to see whether each coarse row is coarsened again for every fine row.
template <std::size_t dim>
static void BM_CoarsenThenRefine(benchmark::State& state)
{
    const auto& lca = fine<dim>();
    run(state,
        [&]
        {
            return samurai::self(lca).on(fine_level<dim>() - 1).on(fine_level<dim>());
        });
    report_fine_rows(state, lca);
}

BENCHMARK(BM_CoarsenThenRefine<1>);
BENCHMARK(BM_CoarsenThenRefine<2>);
BENCHMARK(BM_CoarsenThenRefine<3>);

// --- Expansion of a coarsened set ------------------------------------------------

// expand(self(a).on(L - s), 1), the prediction ghosts of
// MRMesh::update_sub_mesh_impl (s = 1 and 2), against the same expansion of a
// coarse set already built (BM_ExpandCoarsenPrebuilt): does the expansion
// coarsen again for every neighbouring row? Arg: s.
template <std::size_t dim>
static void BM_ExpandCoarsen(benchmark::State& state)
{
    const auto& lca     = fine<dim>();
    const std::size_t s = static_cast<std::size_t>(state.range(0));
    run(state,
        [&]
        {
            return samurai::expand(samurai::self(lca).on(fine_level<dim>() - s), 1);
        });
    report_fine_rows(state, lca);
}

template <std::size_t dim>
static void BM_ExpandCoarsenPrebuilt(benchmark::State& state)
{
    const auto& lca   = fine<dim>();
    const auto coarse = coarsened(lca, static_cast<std::size_t>(state.range(0)));
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

BENCHMARK_MAIN();
