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

// Observation :
// Pourquoi -default est plus lent que _empty_lcl_to_lca en 2D/3D ??
// max_indices et min_indices très couteux : on peut pas faire mieux ??????

constexpr int DEFAULT_X_INTERVALS = 5; // nombre d'intervalles en X pour les cas 2D/3D

///////////////////////////////////////////////////////////////////
// utils

template <unsigned int dim>
auto gen_regular_intervals = [](auto& lcl, int max_index, unsigned int level, int x_intervals = DEFAULT_X_INTERVALS)
{
    int interval_size = 1 << level;       // 2^level
    int spacing       = 1 << (level + 1); // 2^(level+1)

    if constexpr (dim == 1)
    {
        // En 1D on garde le comportement précédent : un intervalle par abscisse.
        for (int x = 0; x < max_index; ++x)
        {
            int start = x * spacing;
            lcl[{}].add_interval({start, start + interval_size});
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
                lcl[{y}].add_interval({start, end});
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
                    lcl[{y, z}].add_interval({start, end});
                }
            }
        }
    }
};

template <unsigned int dim>
auto gen_offset_intervals = [](auto& lcl, int max_index, unsigned int level, int x_intervals = DEFAULT_X_INTERVALS)
{
    int interval_size = 1 << level;       // 2^level
    int spacing       = 1 << (level + 1); // 2^(level+1)

    if constexpr (dim == 1)
    {
        for (int x = 0; x < max_index; ++x)
        {
            int start = x * spacing + interval_size; // Décalage pour être disjoint
            lcl[{}].add_interval({start, start + interval_size});
        }
    }
    else if constexpr (dim == 2)
    {
        int nx = x_intervals;
        for (int x = 0; x < nx; ++x)
        {
            int start = x * spacing + interval_size;
            int end   = start + interval_size;
            for (int y = 0; y < max_index; ++y)
            {
                lcl[{y}].add_interval({start, end});
            }
        }
    }
    else if constexpr (dim == 3)
    {
        int nx = x_intervals;
        for (int x = 0; x < nx; ++x)
        {
            int start = x * spacing + interval_size;
            int end   = start + interval_size;
            for (int y = 0; y < max_index; ++y)
            {
                for (int z = 0; z < max_index; ++z)
                {
                    lcl[{y, z}].add_interval({start, end});
                }
            }
        }
    }
};

template <unsigned int dim>
auto gen_unique_interval = [](auto& lcl, int max_index, unsigned int level, int x_intervals = DEFAULT_X_INTERVALS)
{
    int spacing       = 1 << (level + 1);    // 2^(level+1)
    int interval_size = (max_index)*spacing; // Grande taille proportionnelle à max_index

    if constexpr (dim == 1)
    {
        lcl[{}].add_interval({0, interval_size});
    }
    else if constexpr (dim == 2)
    {
        for (int y = 0; y < max_index; ++y)
        {
            lcl[{y}].add_interval({0, interval_size});
        }
    }
    else if constexpr (dim == 3)
    {
        for (int y = 0; y < max_index; ++y)
        {
            for (int z = 0; z < max_index; ++z)
            {
                lcl[{y, z}].add_interval({0, interval_size});
            }
        }
    }
};

///////////////////////////////////////////////////////////////////

// Mesure : Constructeur par défaut d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_default(benchmark::State& state)
{
    using TInterval = samurai::default_config::interval_t;
    for (auto _ : state)
    {
        auto lcl = samurai::LevelCellArray<dim, TInterval>();
        benchmark::DoNotOptimize(lcl);
    }
}

// Mesure : Constructiuon d'un LevelCellArray à partir d'un LevelCellList vide
template <unsigned int dim>
void LEVELCELLARRAY_empty_lcl_to_lca(benchmark::State& state)
{
    using TInterval = samurai::default_config::interval_t;
    samurai::LevelCellList<dim> lcl;
    for (auto _ : state)
    {
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);
        benchmark::DoNotOptimize(lca);
    }
}

// Mesure : Construction d'un LevelCellArray à partir d'un LevelCellList composé de n intervalles dans une direction
template <unsigned int dim>
void LEVELCELLARRAY_lcl_to_lca(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);

    // Créer un LevelCellArray temporaire pour compter les intervalles
    auto temp_lca        = samurai::LevelCellArray<dim, TInterval>(lcl);
    auto total_intervals = temp_lca.nb_intervals();

    state.counters["Dimension"]       = dim;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);
        benchmark::DoNotOptimize(lca);
    }
}

// Mesure : Récupération de l'intérateur begin d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_begin(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    for (auto _ : state)
    {
        auto begin = lca.begin();
        benchmark::DoNotOptimize(begin);
    }
}

// Mesure : Récupération de l'itéateur end d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_end(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    for (auto _ : state)
    {
        auto end = lca.end();
        benchmark::DoNotOptimize(end);
    }
}

// Mesure : Récupération de la taille d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_shape(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    for (auto _ : state)
    {
        auto shape = lca.shape();
        benchmark::DoNotOptimize(shape);
    }
}

// Mesure : Récupération du nombre d'intervalles dans un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_nb_intervals(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    for (auto _ : state)
    {
        auto nb = lca.nb_intervals();
        benchmark::DoNotOptimize(nb);
    }
}

// Mesure : Récupération du nombre de cellules dans un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_nb_cells(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    auto total_intervals              = lca.nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto nb = lca.nb_cells();
        benchmark::DoNotOptimize(nb);
    }
}

// Mesure : Récupération de la taille de cellule d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_cell_length(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);
    for (auto _ : state)
    {
        auto length = lca.cell_length();
        benchmark::DoNotOptimize(length);
    }
}

// Mesure : Récupération du max d'indice d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_max_indices(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    auto total_intervals              = lca.nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto max = lca.max_indices();
        benchmark::DoNotOptimize(max);
    }
}

// Mesure : Récupération du min d'indide d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_min_indices(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    auto total_intervals              = lca.nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto min = lca.min_indices();
        benchmark::DoNotOptimize(min);
    }
}

// Mesure : Récupération du minmax d'indice d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_minmax_indices(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    auto total_intervals              = lca.nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto minmax = lca.minmax_indices();
        benchmark::DoNotOptimize(minmax);
    }
}

// Mesure : Test d'égalité entre deux LevelCellArrays égaux de taille n
template <unsigned int dim>
void LEVELCELLARRAY_equal(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca  = samurai::LevelCellArray<dim, TInterval>(lcl);
    auto lca2 = samurai::LevelCellArray<dim, TInterval>(lcl);

    auto total_intervals              = lca.nb_intervals() + lca2.nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto is_equal = (lca == lca2);
        benchmark::DoNotOptimize(is_equal);
    }
}

// Mesure : Récupération du niveau bas d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_min_level(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    for (auto _ : state)
    {
        auto min = lca.level();
        benchmark::DoNotOptimize(min);
    }
}

// Mesure : Récupération de l'itérateur reverse begin d'un LevelCellArray
template <unsigned int dim>
void LEVELCELLARRAY_rbegin(benchmark::State& state)
{
    samurai::LevelCellList<dim> lcl;
    using TInterval = samurai::default_config::interval_t;
    int max_index   = static_cast<int>(state.range(0));
    gen_regular_intervals<dim>(lcl, max_index, 0, DEFAULT_X_INTERVALS);
    auto lca = samurai::LevelCellArray<dim, TInterval>(lcl);

    for (auto _ : state)
    {
        auto rbegin = lca.rbegin();
        benchmark::DoNotOptimize(rbegin);
    }
}

// manque les LevelCellList_iterator
BENCHMARK_TEMPLATE(LEVELCELLARRAY_default, 1);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_default, 2);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_default, 3);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_empty_lcl_to_lca, 1);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_empty_lcl_to_lca, 2);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_empty_lcl_to_lca, 3);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_lcl_to_lca, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_lcl_to_lca, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_lcl_to_lca, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_begin, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_begin, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_begin, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_end, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_end, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_end, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_shape, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_shape, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_shape, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_intervals, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_intervals, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_intervals, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_cells, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_cells, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_cells, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_cell_length, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_cell_length, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_cell_length, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_max_indices, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_max_indices, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_max_indices, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_indices, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_indices, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_indices, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_minmax_indices, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_minmax_indices, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_minmax_indices, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_equal, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_equal, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_equal, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_level, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_level, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_level, 3)->Arg(45);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_rbegin, 1)->Arg(10000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_rbegin, 2)->Arg(2000);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_rbegin, 3)->Arg(45);
