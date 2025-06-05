
#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/uniform_mesh.hpp>

// TODO :
// Eviter les maillages aléatoires (biais de répétabilité)

///////////////////////////////////////////////////////////////////
// utils

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

template <unsigned int dim>
auto cell_array_with_n_intervals(int64_t size)
{
    auto cl = cell_list_with_n_intervals<dim>(size);
    samurai::CellArray<dim> ca(cl);
    return ca;
}

//////////////////////////////////////////////////////////////////

// Mesure : Création d'un CellArray par défaut
template <unsigned int dim, unsigned int max_level>
void CELLARRAY_default(benchmark::State& state)
{
    for (auto _ : state)
    {
        samurai::CellArray ca = samurai::CellArray<dim, samurai::default_config::interval_t, max_level>();
    }
}

// Mesure : Création d'un CellArray à partir d'un CellList composé de n intervalles sur une dimension
template <unsigned int dim>
void CELLARRAY_cl_ca_multi(benchmark::State& state)
{
    auto cl = cell_list_with_n_intervals<dim>(state.range(0));
    for (auto _ : state)
    {
        samurai::CellArray<dim> ca(cl);
        benchmark::DoNotOptimize(ca[0]);
    }
}

// Mesure : Récupération du niveau bas d'un CellList
template <unsigned int dim>
void CELLARRAY_min_level(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    cl[state.range(0)][{}].add_interval({0, 1});
    samurai::CellArray<dim> ca(cl);
    for (auto _ : state)
    {
        auto min = ca.min_level();
        benchmark::DoNotOptimize(min);
    }
}

// Mesure : Récupération de l'itérateur begin d'un CellArray
template <unsigned int dim>
void CELLARRAY_begin(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    cl[state.range(0)][{}].add_interval({0, 1});
    samurai::CellArray<dim> ca(cl);
    for (auto _ : state)
    {
        auto begin = ca.begin();
        benchmark::DoNotOptimize(begin);
    }
}

// Mesure : Récupérayion de l'itérateur end d'un CellArray
template <unsigned int dim>
void CELLARRAY_end(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    cl[state.range(0)][{}].add_interval({0, 1});
    samurai::CellArray<dim> ca(cl);
    for (auto _ : state)
    {
        auto end = ca.end();
        benchmark::DoNotOptimize(end);
    }
}

// Mesure : Récupération de l'itérateur reverse begin d'un CellArray
template <unsigned int dim>
void CELLARRAY_rbegin(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    cl[state.range(0)][{}].add_interval({0, 1});
    samurai::CellArray<dim> ca(cl);
    for (auto _ : state)
    {
        auto rbegin = ca.rbegin();
        benchmark::DoNotOptimize(rbegin);
    }
}

// Mesure : Création d'un CellArray 2D à partid d'un CellList composé de n intervalles aléatoires
// Ressemble à CELLARRAY_cl_ca_multi
static void CELLARRAY_CellList2CellArray_2D(benchmark::State& state)
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

BENCHMARK(CELLARRAY_CellList2CellArray_2D)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

// Mesure : Création d'un CellArray 3D à partid d'un CellList composé de n intervalles aléatoires
// Ressemble à CELLARRAY_cl_ca_multi
static void CELLARRAY_CellList2CellArray_3D(benchmark::State& state)
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

BENCHMARK(CELLARRAY_CellList2CellArray_3D)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

// Mesure : Comparaison "=" entre deux CellArray aléatoires
// On ne devrait pas utiliser un CellArray aléatoire.
static void CELLARRAY_equal_2D(benchmark::State& state)
{
    constexpr std::size_t dim = 2;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;
    samurai::CellArray<dim> ca;
    samurai::CellArray<dim> ca2;

    for (std::size_t s = 0; s < state.range(0); ++s)
    {
        auto level = std::experimental::randint(min_level, max_level);
        auto x     = std::experimental::randint(0, (100 << level) - 1);
        auto y     = std::experimental::randint(0, (100 << level) - 1);

        cl[level][{y}].add_point(x);
    }
    ca  = {cl};
    ca2 = {cl};

    for (auto _ : state)
    {
        auto equal = ca == ca2;
        benchmark::DoNotOptimize(equal);
    }
}

BENCHMARK(CELLARRAY_equal_2D)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(CELLARRAY_default, 1, 12);
BENCHMARK_TEMPLATE(CELLARRAY_default, 2, 12);
BENCHMARK_TEMPLATE(CELLARRAY_default, 3, 12);

BENCHMARK_TEMPLATE(CELLARRAY_cl_ca_multi, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(CELLARRAY_cl_ca_multi, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(CELLARRAY_cl_ca_multi, 3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(CELLARRAY_min_level, 1)->DenseRange(0, 15);
BENCHMARK_TEMPLATE(CELLARRAY_min_level, 2)->DenseRange(0, 15);
BENCHMARK_TEMPLATE(CELLARRAY_min_level, 3)->DenseRange(0, 15);

BENCHMARK_TEMPLATE(CELLARRAY_begin, 1)->DenseRange(0, 15);
BENCHMARK_TEMPLATE(CELLARRAY_begin, 2)->DenseRange(0, 15);
BENCHMARK_TEMPLATE(CELLARRAY_begin, 3)->DenseRange(0, 15);

BENCHMARK_TEMPLATE(CELLARRAY_rbegin, 1)->DenseRange(0, 15);
BENCHMARK_TEMPLATE(CELLARRAY_rbegin, 2)->DenseRange(0, 15);
BENCHMARK_TEMPLATE(CELLARRAY_rbegin, 3)->DenseRange(0, 15);

BENCHMARK_TEMPLATE(CELLARRAY_end, 1)->DenseRange(0, 15);
BENCHMARK_TEMPLATE(CELLARRAY_end, 2)->DenseRange(0, 15);
BENCHMARK_TEMPLATE(CELLARRAY_end, 3)->DenseRange(0, 15);
