// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Cost of the set algebra operators at a single level, in 1D, 2D and 3D.
//
// Questions answered, one section each:
//
//   1. Floor: what does traversing a set cost compared to a plain loop over
//      the intervals of its LevelCellArray? (BM_RawLoop vs BM_Self)
//   2. Arity: how does an intersection / union / difference scale with the
//      number of its operands? The operands are translates of one set, the
//      stencil pattern of the library (BM_*Arity), on thick objects (balls)
//      so that the output hardly changes with the arity: read `t/op_itv`,
//      the time per operand interval. BM_IntersectionSameArity removes the
//      translations, BM_IntersectionNestedArity writes the same intersection
//      as a chain of binary ones.
//   3. Expansion and contraction: cost against the width.
//   4. Materialization: building a LevelCellArray from an expression, which
//      the library does wherever it keeps a set (graduation, boundary
//      conditions, mesh construction), against traversing it.
//
// The other inputs are a "front": a thin spherical shell at the finest level
// in 2D and 3D, many small segments in 1D (a 1D shell has two intervals).
// The `t/itv` counter (time per emitted interval) is the figure to compare
// across rows; `itv` and `cells` give the output size.

#include <array>
#include <cstddef>
#include <utility>

#include <benchmark/benchmark.h>

#include "set_algebra_fixtures.hpp"

namespace
{
    using bench::Shape;

    template <std::size_t dim>
    constexpr std::size_t front_level()
    {
        return (dim == 1) ? 20 : ((dim == 2) ? 11 : 8);
    }

    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& front()
    {
        static const auto lca = (dim == 1) ? bench::make_shape<dim>(Shape::balls, front_level<dim>(), 0, 4096)
                                           : bench::make_shape<dim>(Shape::shell, front_level<dim>());
        return lca;
    }

    // Thick objects for the arity benchmarks: translates by a few cells keep
    // most of them, so the output size hardly depends on the arity.
    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& balls()
    {
        static const auto lca = bench::make_shape<dim>(Shape::balls, front_level<dim>(), 0, (dim == 1) ? 4096 : 64);
        return lca;
    }

    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& input(Shape shape)
    {
        static const auto box      = bench::make_shape<dim>(Shape::box, front_level<dim>());
        static const auto ball_set = bench::make_shape<dim>(Shape::balls, front_level<dim>(), 0, 64);
        switch (shape)
        {
            case Shape::box:
                return box;
            case Shape::balls:
                return ball_set;
            default:
                return front<dim>();
        }
    }

    // Every operand differs from the others.
    template <std::size_t dim, std::size_t N>
    auto translations()
    {
        return bench::translations<dim, N>(bench::Reach::growing);
    }

    template <class LCA, std::size_t... Is>
    auto make_same(const LCA& lca, std::index_sequence<Is...>)
    {
        return samurai::intersection(samurai::self(lca), ((void)Is, samurai::self(lca))...);
    }

    // intersection(...intersection(intersection(a, t0 a), t1 a)..., tM a)
    template <std::size_t M, class LCA, class Translations>
    auto make_nested(const LCA& lca, const Translations& t)
    {
        if constexpr (M == 1)
        {
            return samurai::intersection(samurai::self(lca), samurai::translate(lca, t[0]));
        }
        else
        {
            return samurai::intersection(make_nested<M - 1>(lca, t), samurai::translate(lca, t[M - 1]));
        }
    }

    // bench::run, with the time per input interval and per operand interval
    // (`t/op_itv`, input intervals times N): the figure to read when the output
    // size changes with N, as for a difference.
    template <std::size_t dim, class Make>
    void run_nary(benchmark::State& state, const samurai::LevelCellArray<dim>& lca, std::size_t n, Make&& make)
    {
        bench::run(state, std::forward<Make>(make));
        bench::report_input(state, lca);
        state.counters["N"]        = static_cast<double>(n);
        state.counters["t/op_itv"] = benchmark::Counter(static_cast<double>(n * lca[0].size()),
                                                        benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    }

    // op(self(a), N - 1 translates of a), on the balls.
    template <std::size_t dim, std::size_t N, class Op>
    void run_arity(benchmark::State& state, Op&& op)
    {
        const auto& lca = balls<dim>();
        const auto t    = translations<dim, N - 1>();
        run_nary(state,
                 lca,
                 N,
                 [&]
                 {
                     return bench::make_nary(op, lca, t, std::make_index_sequence<N - 1>{});
                 });
    }
} // namespace

// --- 1. Floor ------------------------------------------------------------

// Plain loop over the intervals of the LevelCellArray, no set algebra.
template <std::size_t dim, Shape shape>
static void BM_RawLoop(benchmark::State& state)
{
    const auto& lca = input<dim>(shape);
    bench::Tally tally;
    for (auto _ : state)
    {
        tally = {};
        samurai::for_each_interval(lca,
                                   [&](std::size_t, const auto& i, const auto& index)
                                   {
                                       tally(i, index);
                                   });
        benchmark::DoNotOptimize(tally);
    }
    bench::report(state, tally);
    bench::report_input(state, lca);
}

// The same intervals through self(lca): the fixed cost of the traversal.
template <std::size_t dim, Shape shape>
static void BM_Self(benchmark::State& state)
{
    const auto& lca = input<dim>(shape);
    bench::run(state,
               [&]
               {
                   return samurai::self(lca);
               });
    bench::report_input(state, lca);
}

BENCHMARK(BM_RawLoop<1, Shape::shell>);
BENCHMARK(BM_Self<1, Shape::shell>);
BENCHMARK(BM_RawLoop<2, Shape::shell>);
BENCHMARK(BM_Self<2, Shape::shell>);
BENCHMARK(BM_RawLoop<2, Shape::box>);
BENCHMARK(BM_Self<2, Shape::box>);
BENCHMARK(BM_RawLoop<2, Shape::balls>);
BENCHMARK(BM_Self<2, Shape::balls>);
BENCHMARK(BM_RawLoop<3, Shape::shell>);
BENCHMARK(BM_Self<3, Shape::shell>);
BENCHMARK(BM_RawLoop<3, Shape::box>);
BENCHMARK(BM_Self<3, Shape::box>);
BENCHMARK(BM_RawLoop<3, Shape::balls>);
BENCHMARK(BM_Self<3, Shape::balls>);

// --- 2. Arity --------------------------------------------------------------

// self(a) and N - 1 translates of a.
template <std::size_t dim, std::size_t N>
static void BM_IntersectionArity(benchmark::State& state)
{
    run_arity<dim, N>(state, bench::intersection_op);
}

template <std::size_t dim, std::size_t N>
static void BM_UnionArity(benchmark::State& state)
{
    run_arity<dim, N>(state, bench::union_op);
}

template <std::size_t dim, std::size_t N>
static void BM_DifferenceArity(benchmark::State& state)
{
    run_arity<dim, N>(state, bench::difference_op);
}

// N times the same set: the merge and the row lookups without translation.
template <std::size_t dim, std::size_t N>
static void BM_IntersectionSameArity(benchmark::State& state)
{
    const auto& lca = balls<dim>();
    run_nary(state,
             lca,
             N,
             [&]
             {
                 return make_same(lca, std::make_index_sequence<N - 1>{});
             });
}

// The N-ary intersection of BM_IntersectionArity as N - 1 nested binary ones.
template <std::size_t dim, std::size_t N>
static void BM_IntersectionNestedArity(benchmark::State& state)
{
    const auto& lca = balls<dim>();
    const auto t    = translations<dim, N - 1>();
    run_nary(state,
             lca,
             N,
             [&]
             {
                 return make_nested<N - 1>(lca, t);
             });
}

#define SAMURAI_BENCH_ARITY(NAME, DIM) \
    BENCHMARK(NAME<DIM, 2>);           \
    BENCHMARK(NAME<DIM, 3>);           \
    BENCHMARK(NAME<DIM, 5>);           \
    BENCHMARK(NAME<DIM, 9>);           \
    BENCHMARK(NAME<DIM, 17>);

SAMURAI_BENCH_ARITY(BM_IntersectionArity, 1)
SAMURAI_BENCH_ARITY(BM_IntersectionArity, 2)
SAMURAI_BENCH_ARITY(BM_IntersectionArity, 3)
SAMURAI_BENCH_ARITY(BM_UnionArity, 1)
SAMURAI_BENCH_ARITY(BM_UnionArity, 2)
SAMURAI_BENCH_ARITY(BM_UnionArity, 3)
SAMURAI_BENCH_ARITY(BM_DifferenceArity, 1)
SAMURAI_BENCH_ARITY(BM_DifferenceArity, 2)
SAMURAI_BENCH_ARITY(BM_DifferenceArity, 3)
SAMURAI_BENCH_ARITY(BM_IntersectionSameArity, 2)
SAMURAI_BENCH_ARITY(BM_IntersectionSameArity, 3)
SAMURAI_BENCH_ARITY(BM_IntersectionNestedArity, 2)
SAMURAI_BENCH_ARITY(BM_IntersectionNestedArity, 3)

// --- 3. Expansion and contraction ------------------------------------------

// expand(a, w): the native expansion, which nestedExpand() resolves to.
// Arg: w.
template <std::size_t dim>
static void BM_Expand(benchmark::State& state)
{
    const auto& lca = front<dim>();
    const int w     = static_cast<int>(state.range(0));
    bench::run(state,
               [&]
               {
                   return samurai::expand(lca, w);
               });
}

// contract(a, w): intersection of a with its 2 dim translates by +-w,
// written as dim nested 3-ary intersections. Arg: w.
template <std::size_t dim>
static void BM_Contract(benchmark::State& state)
{
    const auto& lca = front<dim>();
    bench::run(state,
               [&]
               {
                   return samurai::contract(lca, static_cast<std::size_t>(state.range(0)));
               });
}

BENCHMARK(BM_Expand<1>)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_Expand<2>)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_Expand<3>)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_Contract<1>)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK(BM_Contract<2>)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK(BM_Contract<3>)->Arg(1)->Arg(2)->Arg(4);

// --- 4. Materialization ----------------------------------------------------

// LevelCellArray(intersection(a, translate(a, e_0))): what keeping a set
// costs on top of traversing it (compare with BM_IntersectionArity<dim, 2>).
template <std::size_t dim>
static void BM_Materialize(benchmark::State& state)
{
    const auto& lca = front<dim>();
    const auto t    = translations<dim, 1>();
    std::size_t n   = 0;
    for (auto _ : state)
    {
        samurai::LevelCellArray<dim> result(samurai::intersection(lca, samurai::translate(lca, t[0])));
        n = result.nb_cells();
        benchmark::DoNotOptimize(n);
    }
    state.counters["cells"] = static_cast<double>(n);
}

BENCHMARK(BM_Materialize<1>);
BENCHMARK(BM_Materialize<2>);
BENCHMARK(BM_Materialize<3>);

BENCHMARK_MAIN();
