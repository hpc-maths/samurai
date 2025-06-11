
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

// Utils : Créer un cell_list composé de n intervales dans une direction
template <unsigned int dim>
auto cell_list_with_n_intervals(int64_t size)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < size; i++)
    {
        int index = static_cast<int>(i);
        cl[0][{}].add_interval({2 * index, 2 * index + 1});
    }
    return cl;
}

///////////////////////////////////

// Mesure : constructeur CellList par defaut
template <unsigned int dim>
void CELLLIST_default(benchmark::State& state)
{
    for (auto _ : state)
    {
        samurai::CellList<dim> cl = samurai::CellList<dim>();
        benchmark::DoNotOptimize(cl);
    }
}

// Mesure : Création d'un CellList construit par n intervales croissants
template <unsigned int dim>
void CELLLIST_cl_add_interval_end(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (auto _ : state)
    {
        for (int64_t i = 0; i < state.range(0); i++)
        {
            int index = static_cast<int>(i);
            cl[0][{}].add_interval({index, index + 1});
        }
    }
}

// Mesure : Création d'un CellList construit par n intervales décroissants
template <unsigned int dim>
void CELLLIST_cl_add_interval_begin(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (auto _ : state)
    {
        for (int64_t i = state.range(0); i > 0; i--)
        {
            int index = static_cast<int>(i);
            cl[0][{}].add_interval({index, index + 1});
        }
    }
}

// Mesure : Création d'un CellList composé de n fois le même interval
template <unsigned int dim>
void CELLLIST_cl_add_interval_same(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (auto _ : state)
    {
        for (int64_t i = state.range(0); i > 0; i--)
        {
            cl[0][{}].add_interval({0, 1});
        }
    }
}

// Mesure : Création d'un CellList composé de n points (1 seul interval)
template <unsigned int dim>
void CELLLIST_cl_add_point_end(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (auto _ : state)
    {
        for (int64_t i = 0; i < state.range(0); i++)
        {
            int index = static_cast<int>(i);
            cl[0][{}].add_point({index});
        }
    }
}

// Mesure : Création de n intervales aléatoires - permet de controler le cout du générateur d'aléatoire pour le prochain benchmark
static void CELLLIST_CellListConstruction_2D_rand_control(benchmark::State& state)
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

            benchmark::DoNotOptimize(level);
            benchmark::DoNotOptimize(x);
            benchmark::DoNotOptimize(y);
        }
    }
}

BENCHMARK(CELLLIST_CellListConstruction_2D_rand_control)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

// Mesure : Création d'un CellList composé de n intervales aléatoires en 2D
static void CELLLIST_CellListConstruction_2D(benchmark::State& state)
{
    constexpr std::size_t dim = 2;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    for (auto _ : state)
    {
        samurai::CellList<dim> cl;
        for (std::size_t s = 0; s < state.range(0); ++s)
        {
            auto level = std::experimental::randint(min_level, max_level);
            auto x     = std::experimental::randint(0, (100 << level) - 1);
            auto y     = std::experimental::randint(0, (100 << level) - 1);

            cl[level][{y}].add_point(x);
        }
    }
}

BENCHMARK(CELLLIST_CellListConstruction_2D)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

// Mesure : Création d'un CellList composé de n intervales aléaoires en 3D
static void CELLLIST_CellListConstruction_3D(benchmark::State& state)
{
    constexpr std::size_t dim = 3;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    for (auto _ : state)
    {
        samurai::CellList<dim> cl;
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

BENCHMARK(CELLLIST_CellListConstruction_3D)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

// Not a good way to benchmark
// because the benchmark result depends on the option --benchmark_min_time
// that's because cl is not reinitialized for each state.range measure.
// thus the insertion in std::map is slower because larger and larger
// I provide a fix that puts cl in for state loop.
// We measure the cors of its declaration, you have to substract to have what you really want :-)
/**
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
**/

BENCHMARK_TEMPLATE(CELLLIST_default, 1);
BENCHMARK_TEMPLATE(CELLLIST_default, 2);
BENCHMARK_TEMPLATE(CELLLIST_default, 3);

BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_end, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_end, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_end, 3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_begin, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_same, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(CELLLIST_cl_add_point_end, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(CELLLIST_cl_add_point_end, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(CELLLIST_cl_add_point_end, 3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
