
#include <benchmark/benchmark.h>

#include <xtensor/xfixed.hpp>

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

// TODO :

/////////////////////////////////////////////////////////////////////////////
/// utils
template <unsigned int dim>
auto unitary_box()
{
    using value_t = samurai::default_config::value_t;
    using point_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;
    point_t point1;
    point_t point2;
    if constexpr (dim == 1)
    {
        point1 = {0};
        point2 = {1};
    }
    if constexpr (dim == 2)
    {
        point1 = {0, 0};
        point2 = {1, 1};
    }
    if constexpr (dim == 3)
    {
        point1 = {0, 0, 0};
        point2 = {1, 1, 1};
    }

    samurai::Box<double, dim> box = samurai::Box<double, dim>(point1, point2);
    return box;
}

////////////////////////////////////////////////////////////////////////////

// Mesure : Création d'une Box uniforme de taille de coté n
template <unsigned int dim>
void MESH_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    for (auto _ : state)
    {
        auto mesh = samurai::UniformMesh<Config>(box, state.range(0));
    }
}

BENCHMARK_TEMPLATE(MESH_uniform, 1)->DenseRange(1, 16);
BENCHMARK_TEMPLATE(MESH_uniform, 2)->DenseRange(1, 14);
BENCHMARK_TEMPLATE(MESH_uniform, 3)->DenseRange(1, 9);
