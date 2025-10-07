#include <benchmark/benchmark.h>
#include <cmath>

#include <xtensor/xfixed.hpp>

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

constexpr int DEFAULT_X_INTERVALS = 5; // nombre d'intervalles en X pour les cas 2D/3D

template <unsigned int dim>
auto gen_regular_intervals = [](int max_index, unsigned int level = 0, int x_intervals = DEFAULT_X_INTERVALS)
{
    samurai::CellList<dim> cl;

    int interval_size = 1 << level;       // 2^level
    int spacing       = 1 << (level + 1); // 2^(level+1)

    if constexpr (dim == 1)
    {
        // En 1D on garde le comportement précédent : un intervalle par abscisse.
        for (int x = 0; x < max_index; ++x)
        {
            int start = x * spacing;
            cl[level][{}].add_interval({start, start + interval_size});
        }
    }
    else if constexpr (dim == 2)
    {
        int nx = x_intervals;
        for (int x = 0; x < nx; ++x)
        {
            int start = x * spacing;
            int end   = start + interval_size;
            for (int y = 0; y < max_index; ++y)
            {
                xt::xtensor_fixed<int, xt::xshape<1>> coord{y};
                cl[level][coord].add_interval({start, end});
            }
        }
    }
    else if constexpr (dim == 3)
    {
        int nx = x_intervals;
        for (int x = 0; x < nx; ++x)
        {
            int start = x * spacing;
            int end   = start + interval_size;
            for (int y = 0; y < max_index; ++y)
            {
                for (int z = 0; z < max_index; ++z)
                {
                    xt::xtensor_fixed<int, xt::xshape<2>> coord{y, z};
                    cl[level][coord].add_interval({start, end});
                }
            }
        }
    }

    return cl;
};

template <unsigned int dim>
auto cell_array_with_n_intervals(int max_index)
{
    auto cl = gen_regular_intervals<dim>(max_index, 0, DEFAULT_X_INTERVALS);
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
    int max_index = static_cast<int>(state.range(0));
    auto cl       = gen_regular_intervals<dim>(max_index, 0, DEFAULT_X_INTERVALS);

    // Calculer le nombre d'intervalles selon la nouvelle logique
    std::size_t nb_intervals;
    if constexpr (dim == 1)
    {
        nb_intervals = max_index;
    }
    else if constexpr (dim == 2)
    {
        nb_intervals = DEFAULT_X_INTERVALS * max_index;
    }
    else if constexpr (dim == 3)
    {
        nb_intervals = DEFAULT_X_INTERVALS * max_index * max_index;
    }

    for (auto _ : state)
    {
        samurai::CellArray<dim> ca(cl);
        benchmark::DoNotOptimize(ca[0]);
    }

    // Ajouter les compteurs
    state.counters["nb_intervals"] = nb_intervals;
    state.counters["ns/interval"]  = benchmark::Counter(nb_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["dim"]          = dim;

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(nb_intervals));
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
    for (int level = 0; level <= state.range(0); ++level)
    {
        cl[level][{}].add_interval({0, 1});
    }
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
    for (int level = 0; level <= state.range(0); ++level)
    {
        cl[level][{}].add_interval({0, 1});
    }
    samurai::CellArray<dim> ca(cl);
    for (auto _ : state)
    {
        auto end = ca.end();
        benchmark::DoNotOptimize(end);
    }
}

BENCHMARK_TEMPLATE(CELLARRAY_default, 1, 12);
BENCHMARK_TEMPLATE(CELLARRAY_default, 2, 12);
BENCHMARK_TEMPLATE(CELLARRAY_default, 3, 12);

BENCHMARK_TEMPLATE(CELLARRAY_cl_ca_multi, 1)->RangeMultiplier(8)->Range(1 << 1, 10000);
BENCHMARK_TEMPLATE(CELLARRAY_cl_ca_multi, 2)->RangeMultiplier(4)->Range(1 << 1, 2000);
BENCHMARK_TEMPLATE(CELLARRAY_cl_ca_multi, 3)->RangeMultiplier(2)->Range(1 << 1, 45);

BENCHMARK_TEMPLATE(CELLARRAY_min_level, 1)->Arg(15);
BENCHMARK_TEMPLATE(CELLARRAY_min_level, 2)->Arg(15);
BENCHMARK_TEMPLATE(CELLARRAY_min_level, 3)->Arg(15);

BENCHMARK_TEMPLATE(CELLARRAY_begin, 1)->Arg(15);
BENCHMARK_TEMPLATE(CELLARRAY_begin, 2)->Arg(15);
BENCHMARK_TEMPLATE(CELLARRAY_begin, 3)->Arg(15);

BENCHMARK_TEMPLATE(CELLARRAY_end, 1)->Arg(15);
BENCHMARK_TEMPLATE(CELLARRAY_end, 2)->Arg(15);
BENCHMARK_TEMPLATE(CELLARRAY_end, 3)->Arg(15);
