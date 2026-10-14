// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Cost of the set operators at a single level, in 1D, 2D and 3D: the fixed
// cost of a traversal, the arity of intersection and union, and expansion.
//
// Inputs: a front (a thin spherical shell in 2D and 3D, many small segments
// in 1D, where a shell has only two intervals), a box, and balls for the
// arity benchmarks.

#include <array>
#include <cstddef>
#include <utility>

#include <benchmark/benchmark.h>

#include <samurai/algorithm.hpp>

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
        static const auto lca = bench::make_front<dim>(front_level<dim>());
        return lca;
    }

    // Thick objects: translates by a few cells keep most of their cells, so
    // the output of the arity benchmarks hardly depends on the arity.
    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& balls()
    {
        static const auto lca = bench::make_shape<dim>(Shape::balls, front_level<dim>(), 0, (dim == 1) ? 4096 : 64);
        return lca;
    }

    // Shape::shell stands for the front.
    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& input(Shape shape)
    {
        static const auto box = bench::make_shape<dim>(Shape::box, front_level<dim>());
        return (shape == Shape::box) ? box : front<dim>();
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

    // `t/op_itv` is the time per operand interval (input intervals times N),
    // the figure that compares arities.
    template <std::size_t dim, class Make>
    void run_nary(benchmark::State& state, const samurai::LevelCellArray<dim>& lca, std::size_t n, Make&& make)
    {
        bench::run(state, std::forward<Make>(make));
        bench::report_input(state, lca);
        state.counters["N"]        = static_cast<double>(n);
        state.counters["t/op_itv"] = benchmark::Counter(static_cast<double>(n * lca[0].size()),
                                                        benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    }

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

// --- Fixed cost of a traversal ---------------------------------------------

// The floor below which no expression can go: BM_Self against BM_RawLoop on
// the same LevelCellArray is the cost of the traversal machinery itself.
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
BENCHMARK(BM_RawLoop<3, Shape::shell>);
BENCHMARK(BM_Self<3, Shape::shell>);
BENCHMARK(BM_RawLoop<3, Shape::box>);
BENCHMARK(BM_Self<3, Shape::box>);

// --- Arity of the N-ary operators ---------------------------------------------

// How the cost per operand interval grows with the number of operands, when
// they are translates of one set as in the library's stencils.
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

// Control for BM_IntersectionArity: N copies of the same set, whose
// intervals are in phase. If the cost per operand interval stays flat here,
// the growth in BM_IntersectionArity comes from out-of-phase operands.
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

// The intersection of BM_IntersectionArity written as N - 1 nested binary
// ones: does nesting avoid the growth with N?
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
SAMURAI_BENCH_ARITY(BM_IntersectionSameArity, 2)
SAMURAI_BENCH_ARITY(BM_IntersectionSameArity, 3)
SAMURAI_BENCH_ARITY(BM_IntersectionNestedArity, 2)
SAMURAI_BENCH_ARITY(BM_IntersectionNestedArity, 3)

// --- Expansion ------------------------------------------------------------------

// expand(a, w), which nestedExpand() resolves to: how the cost grows with the
// width, against BM_Self on the same front.
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

BENCHMARK(BM_Expand<1>)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_Expand<2>)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_Expand<3>)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

BENCHMARK_MAIN();
