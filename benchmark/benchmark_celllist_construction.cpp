#include <benchmark/benchmark.h>
#include <experimental/random>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

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
            auto level = std::experimental::randint(min_level, max_level);
            auto x     = std::experimental::randint(0, (100 << level) - 1);
            auto y     = std::experimental::randint(0, (100 << level) - 1);

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
            auto level = std::experimental::randint(min_level, max_level);
            auto x     = std::experimental::randint(0, (100 << level) - 1);
            auto y     = std::experimental::randint(0, (100 << level) - 1);
            auto z     = std::experimental::randint(0, (100 << level) - 1);

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
        auto level = std::experimental::randint(min_level, max_level);
        auto x     = std::experimental::randint(0, (100 << level) - 1);
        auto y     = std::experimental::randint(0, (100 << level) - 1);

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
        auto level = std::experimental::randint(min_level, max_level);
        auto x     = std::experimental::randint(0, (100 << level) - 1);
        auto y     = std::experimental::randint(0, (100 << level) - 1);
        auto z     = std::experimental::randint(0, (100 << level) - 1);

        cl[level][{y, z}].add_point(x);
    }

    for (auto _ : state)
    {
        ca = {cl};
    }
}

BENCHMARK(BM_CellList2CellArray_3D)->Range(8, 8 << 18);
