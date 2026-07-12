// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Cost of populating a CellList by random point insertion, and of the
// CellList -> CellArray conversion, in 2D and 3D, for a range of insertion
// counts (state.range(0)).
//
// CellList is the intermediate structure every mesh-construction path (box
// initialization, MR adaptation, restart loading) writes into before it is
// compacted into the interval-based CellArray. This isolates the cost of
// that write-then-compact step from the rest of mesh construction; it does
// not measure a realistic access pattern (points are independent and
// uniformly distributed across levels, not spatially coherent as in an
// adapted mesh).

#include <random>

#include <benchmark/benchmark.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

namespace
{
    // Fixed-seed uniform integer draw, reproducible across runs.
    template <class T>
    T randint(T lo, T hi)
    {
        static std::mt19937 gen{42};
        std::uniform_int_distribution<T> dist(lo, hi);
        return dist(gen);
    }
}

static void BM_CellListConstruction_2D(benchmark::State& state)
{
    constexpr std::size_t dim = 2;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;

    for (auto _ : state)
    {
        for (std::size_t s = 0; s < state.range(0); ++s)
        {
            auto level = randint(min_level, max_level);
            auto x     = randint(0, (100 << level) - 1);
            auto y     = randint(0, (100 << level) - 1);

            cl[level][{y}].add_point(x);
        }
    }
}

BENCHMARK(BM_CellListConstruction_2D)->Range(8, 8 << 18);

static void BM_CellListConstruction_3D(benchmark::State& state)
{
    constexpr std::size_t dim = 3;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;

    for (auto _ : state)
    {
        for (std::size_t s = 0; s < state.range(0); ++s)
        {
            auto level = randint(min_level, max_level);
            auto x     = randint(0, (100 << level) - 1);
            auto y     = randint(0, (100 << level) - 1);
            auto z     = randint(0, (100 << level) - 1);

            cl[level][{y, z}].add_point(x);
        }
    }
}

BENCHMARK(BM_CellListConstruction_3D)->Range(8, 8 << 18);

static void BM_CellList2CellArray_2D(benchmark::State& state)
{
    constexpr std::size_t dim = 2;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;
    samurai::CellArray<dim> ca;

    for (std::size_t s = 0; s < state.range(0); ++s)
    {
        auto level = randint(min_level, max_level);
        auto x     = randint(0, (100 << level) - 1);
        auto y     = randint(0, (100 << level) - 1);

        cl[level][{y}].add_point(x);
    }

    for (auto _ : state)
    {
        ca = {cl};
    }
}

BENCHMARK(BM_CellList2CellArray_2D)->Range(8, 8 << 18);

static void BM_CellList2CellArray_3D(benchmark::State& state)
{
    constexpr std::size_t dim = 3;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;
    samurai::CellArray<dim> ca;

    for (std::size_t s = 0; s < state.range(0); ++s)
    {
        auto level = randint(min_level, max_level);
        auto x     = randint(0, (100 << level) - 1);
        auto y     = randint(0, (100 << level) - 1);
        auto z     = randint(0, (100 << level) - 1);

        cl[level][{y, z}].add_point(x);
    }

    for (auto _ : state)
    {
        ca = {cl};
    }
}

BENCHMARK(BM_CellList2CellArray_3D)->Range(8, 8 << 18);

BENCHMARK_MAIN();
