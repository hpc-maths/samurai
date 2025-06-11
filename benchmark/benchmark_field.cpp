
#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/uniform_mesh.hpp>

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
void MESH_default(benchmark::State& state)
{
    using Config = samurai::amr::Config<dim>;
    for (auto _ : state)
    {
        auto mesh = samurai::amr::Mesh<Config>();
        benchmark::DoNotOptimize(mesh);
    }
}

// probablement mal fait puisque add interval au niveau 0 sur des aussi grandes valeurs ...
template <unsigned int dim>
void MESH_cl(benchmark::State& state)
{
    using Config   = samurai::amr::Config<dim>;
    auto cl        = cell_list_with_n_intervals<dim>(state.range(0));
    auto min_level = 1;
    auto max_level = 7;
    for (auto _ : state)
    {
        auto mesh = samurai::amr::Mesh<Config>(cl, min_level, max_level);
        benchmark::DoNotOptimize(mesh);
    }
}

BENCHMARK_TEMPLATE(MESH_default, 1);
BENCHMARK_TEMPLATE(MESH_default, 2);
BENCHMARK_TEMPLATE(MESH_default, 3);

BENCHMARK_TEMPLATE(MESH_cl, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(MESH_cl, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 8);
BENCHMARK_TEMPLATE(MESH_cl, 3)->RangeMultiplier(2)->Range(1 << 1, 1 << 7);
