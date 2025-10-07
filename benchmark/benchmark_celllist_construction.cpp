#include <benchmark/benchmark.h>

#include <xtensor/xfixed.hpp>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

////////////////////////////////////////////////////////////
/// Générateur d'intervalles réguliers (adapté de benchmark_search.cpp)

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

// Ancien générateur simple (1D seulement)
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

// Générateur avec points individuels (1D seulement) - stride de 2
template <unsigned int dim>
auto cell_list_with_n_points(int64_t size)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < size; i++)
    {
        int index = static_cast<int>(i);
        cl[0][{}].add_point(2 * index); // stride de 2: 0, 2, 4, 6, etc.
    }
    return cl;
}

////////////////////////////////////////////////////////////
/// Fonction utilitaire pour compter les intervalles

template <unsigned int dim>
std::size_t count_intervals(const samurai::CellList<dim>& cl)
{
    std::size_t count = 0;
    samurai::CellArray<dim> ca(cl);
    samurai::for_each_interval(ca,
                               [&](std::size_t, const auto&, const auto&)
                               {
                                   count++;
                               });
    return count;
}

///////////////////////////////////

// Mesure : constructeur CellList par défaut
template <unsigned int dim>
void CELLLIST_default(benchmark::State& state)
{
    for (auto _ : state)
    {
        samurai::CellList<dim> cl;
        benchmark::DoNotOptimize(cl);
    }

    state.counters["dimension"] = static_cast<double>(dim);
    state.counters["intervals"] = 0;
}

// Mesure : Construction CellList avec intervalles réguliers (ordre décroissant - begin)

// Mesure : Construction CellList avec le même intervalle répété (same)

// Mesure : Construction CellList avec intervalles réguliers
template <unsigned int dim>
void CELLLIST_add_interval_begin(benchmark::State& state)
{
    int max_index = static_cast<int>(state.range(0));
    // Calculer une seule fois pour les métriques
    auto cl_sample              = gen_regular_intervals<dim>(max_index, 0, DEFAULT_X_INTERVALS);
    std::size_t total_intervals = count_intervals<dim>(cl_sample);

    for (auto _ : state)
    {
        auto cl = gen_regular_intervals<dim>(max_index, 0, DEFAULT_X_INTERVALS);
        benchmark::DoNotOptimize(cl);
    }

    state.counters["dimension"]   = static_cast<double>(dim);
    state.counters["intervals"]   = static_cast<double>(total_intervals);
    state.counters["ns/interval"] = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_intervals));
}

// Mesure : Copie de CellList par opérateur d'assignation
template <unsigned int dim>
void CELLLIST_copy_assignment(benchmark::State& state)
{
    int max_index               = static_cast<int>(state.range(0));
    auto source_cl              = gen_regular_intervals<dim>(max_index, 0, DEFAULT_X_INTERVALS);
    std::size_t total_intervals = count_intervals<dim>(source_cl);

    for (auto _ : state)
    {
        samurai::CellList<dim> copied_cl;
        copied_cl = source_cl;
        benchmark::DoNotOptimize(copied_cl);
    }

    state.counters["dimension"]   = static_cast<double>(dim);
    state.counters["intervals"]   = static_cast<double>(total_intervals);
    state.counters["ns/interval"] = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_intervals));
}

////////////////////////////////////////////////////////////
/// Enregistrement des benchmarks

// Constructeur par défaut
BENCHMARK_TEMPLATE(CELLLIST_default, 1);
BENCHMARK_TEMPLATE(CELLLIST_default, 2);
BENCHMARK_TEMPLATE(CELLLIST_default, 3);

// Générateur avec intervalles réguliers (toutes dimensions) - Ajusté pour ~10k intervalles max
BENCHMARK_TEMPLATE(CELLLIST_add_interval_begin, 1)->RangeMultiplier(64)->Range(1 << 1, 10000);
BENCHMARK_TEMPLATE(CELLLIST_add_interval_begin, 2)->RangeMultiplier(8)->Range(1 << 1, 2000);
BENCHMARK_TEMPLATE(CELLLIST_add_interval_begin, 3)->RangeMultiplier(4)->Range(1 << 1, 45);

BENCHMARK_TEMPLATE(CELLLIST_copy_assignment, 1)->RangeMultiplier(64)->Range(1 << 1, 10000);
BENCHMARK_TEMPLATE(CELLLIST_copy_assignment, 2)->RangeMultiplier(8)->Range(1 << 1, 2000);
BENCHMARK_TEMPLATE(CELLLIST_copy_assignment, 3)->RangeMultiplier(4)->Range(1 << 1, 45);
