// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Cost of reading a set at another level with .on(l), in 1D, 2D and 3D. The
// coarsening is the suspect: for every coarse row it gathers the intervals of
// all the fine rows it covers (subset_utils::transform_to_loi), and nothing
// keeps the result between two evaluations.
//
// Input: a front at the fine level L, the first argument of every benchmark
// (bench::synthetic_levels): a thin shell in 2D and 3D, many segments in 1D.
// `t/in_row` is the time per fine row, the unit of work of the coarsening.

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "set_algebra_fixtures.hpp"

namespace
{
    using bench::make_front;
    using bench::run;
    using bench::synthetic_level;

    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& fine(const benchmark::State& state)
    {
        return bench::front<dim>(synthetic_level(state));
    }

    template <std::size_t dim>
    samurai::LevelCellArray<dim> coarsened(const samurai::LevelCellArray<dim>& lca, std::size_t s)
    {
        return samurai::LevelCellArray<dim>(samurai::self(lca).on(lca.level() - s));
    }

    // Args: (L, s) for each s of `differences` that leaves a level of at
    // least 4 cells across, L - s >= 2.
    template <std::size_t dim>
    void difference_args(benchmark::internal::Benchmark* b, std::initializer_list<int64_t> differences)
    {
        const std::vector<int64_t> all(differences);
        bench::level_args<dim>(b,
                               [&](std::size_t level)
                               {
                                   std::vector<std::vector<int64_t>> args;
                                   for (const int64_t s : all)
                                   {
                                       if (static_cast<int64_t>(level) - s >= 2)
                                       {
                                           args.push_back({s});
                                       }
                                   }
                                   return args;
                               });
    }

    template <std::size_t dim>
    void report_fine_rows(benchmark::State& state, const samurai::LevelCellArray<dim>& lca)
    {
        bench::report_input(state, lca);
        state.counters["t/in_row"] = benchmark::Counter(static_cast<double>(bench::count_rows(lca)),
                                                        benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    }

    // The difference of levels s of Args (L, s).
    inline std::size_t level_difference(const benchmark::State& state)
    {
        return static_cast<std::size_t>(state.range(1));
    }

    // Runs make(a, L - s) on the front a at L, for Args (L, s).
    template <std::size_t dim, class Make>
    void run_on_fine(benchmark::State& state, Make&& make)
    {
        const auto& lca     = fine<dim>(state);
        const std::size_t s = level_difference(state);
        run(state,
            [&]
            {
                return make(lca, lca.level() - s);
            });
        report_fine_rows(state, lca);
    }

    // Runs make(c) on the front at L coarsened to L - s beforehand, for Args
    // (L, s).
    template <std::size_t dim, class Make>
    void run_on_coarsened(benchmark::State& state, Make&& make)
    {
        const auto& lca   = fine<dim>(state);
        const auto coarse = coarsened(lca, level_difference(state));
        run(state,
            [&]
            {
                return make(coarse);
            });
        report_fine_rows(state, lca);
    }

    // The front at L - s, for Args (L, s).
    template <std::size_t dim>
    samurai::LevelCellArray<dim> coarse_front(const benchmark::State& state)
    {
        return make_front<dim>(synthetic_level(state) - level_difference(state));
    }
} // namespace

// --- Coarsening against the level difference --------------------------------

// The cost of self(a).on(L - s) per fine row as s grows, compared with
// traversing the coarse set already built (BM_CoarsenPrebuilt) and with
// building it once (BM_CoarsenBuild): the gain a cached coarse set would
// bring. Args: (L, s).
template <std::size_t dim>
static void BM_Coarsen(benchmark::State& state)
{
    run_on_fine<dim>(state,
                     [](const auto& lca, std::size_t level)
                     {
                         return samurai::self(lca).on(level);
                     });
}

template <std::size_t dim>
static void BM_CoarsenPrebuilt(benchmark::State& state)
{
    run_on_coarsened<dim>(state,
                          [](const auto& coarse)
                          {
                              return samurai::self(coarse);
                          });
}

template <std::size_t dim>
static void BM_CoarsenBuild(benchmark::State& state)
{
    const auto& lca     = fine<dim>(state);
    const std::size_t s = level_difference(state);
    for (auto _ : state)
    {
        auto coarse = coarsened(lca, s);
        benchmark::DoNotOptimize(coarse);
    }
    report_fine_rows(state, lca);
}

namespace
{
    template <std::size_t dim>
    void coarsen_args(benchmark::internal::Benchmark* b)
    {
        difference_args<dim>(b, {1, 2, 3, 4, 6});
    }
} // namespace

#define SAMURAI_BENCH_COARSEN(DIM)                                \
    BENCHMARK(BM_Coarsen<DIM>)->Apply(coarsen_args<DIM>);         \
    BENCHMARK(BM_CoarsenPrebuilt<DIM>)->Apply(coarsen_args<DIM>); \
    BENCHMARK(BM_CoarsenBuild<DIM>)->Apply(coarsen_args<DIM>);

SAMURAI_BENCH_COARSEN(1)
SAMURAI_BENCH_COARSEN(2)
SAMURAI_BENCH_COARSEN(3)

// --- Refinement against the level difference --------------------------------

// Control for the coarsening: self(coarse).on(L) against the refined set
// already built (BM_RefinePrebuilt). The coarse front lives at L - s, so the
// output covers the same region at the same resolution for every s. Args:
// (L, s).
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
    const std::size_t level = synthetic_level(state);
    const auto coarse       = coarse_front<dim>(state);
    run(state,
        [&]
        {
            return samurai::self(coarse).on(level);
        });
    bench::report_input(state, coarse);
}

template <std::size_t dim>
static void BM_RefinePrebuilt(benchmark::State& state)
{
    const auto coarse = coarse_front<dim>(state);
    const samurai::LevelCellArray<dim> refined(samurai::self(coarse).on(synthetic_level(state)));
    run(state,
        [&]
        {
            return samurai::self(refined);
        });
    bench::report_input(state, coarse);
}

namespace
{
    template <std::size_t dim>
    void refine_args(benchmark::internal::Benchmark* b)
    {
        if constexpr (dim < 3)
        {
            difference_args<dim>(b, {1, 2, 4, 6});
        }
        else
        {
            difference_args<dim>(b, {1, 2, 4});
        }
    }
} // namespace

BENCHMARK(BM_Refine<1>)->Apply(refine_args<1>);
BENCHMARK(BM_RefinePrebuilt<1>)->Apply(refine_args<1>);
BENCHMARK(BM_Refine<2>)->Apply(refine_args<2>);
BENCHMARK(BM_RefinePrebuilt<2>)->Apply(refine_args<2>);
BENCHMARK(BM_Refine<3>)->Apply(refine_args<3>);
BENCHMARK(BM_RefinePrebuilt<3>)->Apply(refine_args<3>);

// --- Coarsening of a composite expression -----------------------------------

namespace
{
    template <std::size_t dim, std::size_t N, bool coarsen>
    void run_composite(benchmark::State& state)
    {
        const auto& lca = fine<dim>(state);
        const auto t    = bench::translations<dim, N - 1>(bench::Reach::unit);
        run(state,
            [&]
            {
                auto set = bench::make_nary(bench::intersection_op, lca, t, std::make_index_sequence<N - 1>{});
                if constexpr (coarsen)
                {
                    return set.on(lca.level() - 1);
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

#define SAMURAI_BENCH_COMPOSITE(DIM)                                       \
    BENCHMARK(BM_Composite<DIM, 2>)->Apply(bench::level_args<DIM>);        \
    BENCHMARK(BM_CompositeCoarsen<DIM, 2>)->Apply(bench::level_args<DIM>); \
    BENCHMARK(BM_Composite<DIM, 3>)->Apply(bench::level_args<DIM>);        \
    BENCHMARK(BM_CompositeCoarsen<DIM, 3>)->Apply(bench::level_args<DIM>); \
    BENCHMARK(BM_Composite<DIM, 5>)->Apply(bench::level_args<DIM>);        \
    BENCHMARK(BM_CompositeCoarsen<DIM, 5>)->Apply(bench::level_args<DIM>);

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
    const auto& lca = fine<dim>(state);
    run(state,
        [&]
        {
            return samurai::self(lca).on(lca.level() - 1).on(lca.level());
        });
    report_fine_rows(state, lca);
}

BENCHMARK(BM_CoarsenThenRefine<1>)->Apply(bench::level_args<1>);
BENCHMARK(BM_CoarsenThenRefine<2>)->Apply(bench::level_args<2>);
BENCHMARK(BM_CoarsenThenRefine<3>)->Apply(bench::level_args<3>);

// --- Expansion of a coarsened set ------------------------------------------------

// expand(self(a).on(L - s), 1), the prediction ghosts of
// MRMesh::update_sub_mesh_impl (s = 1 and 2), against the same expansion of a
// coarse set already built (BM_ExpandCoarsenPrebuilt): does the expansion
// coarsen again for every neighbouring row? Args: (L, s).
template <std::size_t dim>
static void BM_ExpandCoarsen(benchmark::State& state)
{
    run_on_fine<dim>(state,
                     [](const auto& lca, std::size_t level)
                     {
                         return samurai::expand(samurai::self(lca).on(level), 1);
                     });
}

template <std::size_t dim>
static void BM_ExpandCoarsenPrebuilt(benchmark::State& state)
{
    run_on_coarsened<dim>(state,
                          [](const auto& coarse)
                          {
                              return samurai::expand(coarse, 1);
                          });
}

namespace
{
    template <std::size_t dim>
    void expand_coarsen_args(benchmark::internal::Benchmark* b)
    {
        difference_args<dim>(b, {1, 2});
    }
} // namespace

BENCHMARK(BM_ExpandCoarsen<1>)->Apply(expand_coarsen_args<1>);
BENCHMARK(BM_ExpandCoarsenPrebuilt<1>)->Apply(expand_coarsen_args<1>);
BENCHMARK(BM_ExpandCoarsen<2>)->Apply(expand_coarsen_args<2>);
BENCHMARK(BM_ExpandCoarsenPrebuilt<2>)->Apply(expand_coarsen_args<2>);
BENCHMARK(BM_ExpandCoarsen<3>)->Apply(expand_coarsen_args<3>);
BENCHMARK(BM_ExpandCoarsenPrebuilt<3>)->Apply(expand_coarsen_args<3>);

BENCHMARK_MAIN();
