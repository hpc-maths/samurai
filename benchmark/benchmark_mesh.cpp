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

// Mesure : Allocation d'un champ 1D d'un maillage uniforme de taille de coté n
template <unsigned int dim>
void FIELD_make_field_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    for (auto _ : state)
    {
        auto u = make_field<double, 1>("u", mesh);
    }
}

// Mesure : Remplissage d'un champ 1D de taille de coté n
template <unsigned int dim>
void FIELD_fill_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field<double, 1>("u", mesh);
    for (auto _ : state)
    {
        u.fill(1.0);
    }
}

// Mesure ; Remplissage d'un champ 1D de taille de coté n ,en utilisant for_each_cell (mesure d'overhead)
template <unsigned int dim>
void FIELD_for_each_cell_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field<double, 1>("u", mesh);
    for (auto _ : state)
    {
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          u[cell] = 1.0;
                      });
    }
}

// Mesure : Test d'égalité entre deux Fields

// Weird : MPI issue ??? wtf ???
template <unsigned int dim>
void FIELD_equal_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field<double, 1>("u", mesh);
    auto v                        = make_field<double, 1>("v", mesh);
    u.fill(1.0);
    for (auto _ : state)
    {
        v = u;
    }
}

// Mesure : Ajout d'un scalaire à un champ par broadcasting
template <unsigned int dim>
void FIELD_add_scalar_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field<double, 1>("u", mesh);
    u.fill(1.0);
    auto v = make_field<double, 1>("v", mesh);
    for (auto _ : state)
    {
        v = u + 2.0;
        benchmark::DoNotOptimize(v[0]);
    }
}

// Mesure : Ajout d'un scalaire à un champ par "for_each_cell"
template <unsigned int dim>
void FIELD_add_scalar_for_each_cell_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field<double, 1>("u", mesh);
    u.fill(1.0);
    auto v = make_field<double, 1>("v", mesh);
    for (auto _ : state)
    {
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          v[cell] = u[cell] + 1.0;
                      });
    }
}

/**
template <unsigned int dim>
void FIELD_add_scalar_for_each_interval_uniform(benchmark::State& state){
        samurai::Box<double, dim> box = unitary_box<dim>() ;
        using Config = samurai::UniformConfig<dim> ;
        auto mesh = samurai::UniformMesh<Config>(box, state.range(0));
        auto u = make_field<double, 1>("u", mesh) ;
        u.fill(1.0) ;
        auto v = make_field<double, 1>("v", mesh) ;
    v.fill(0.0) ;
        for (auto _ : state){
                for_each_cell(mesh,
                                [&](std::size_t level, const auto& interval, const auto& index)
                                {
                    auto i = interval ;
                    auto j = index[0] ;
                                        v(level, i, j) = 1.0 ;
                                });
        }
}
**/

// Mesure : Somme de deux champs 1D par "for_each_cell"
template <unsigned int dim>
void FIELD_add_for_each_cell_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field<double, 1>("u", mesh);
    u.fill(1.0);
    auto v = make_field<double, 1>("v", mesh);
    v.fill(1.0);
    auto w = make_field<double, 1>("w", mesh);
    w.fill(1.0);

    for (auto _ : state)
    {
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          w[cell] = u[cell] + v[cell];
                      });
    }
}

// Mesure : Somme de deux champs 1D par expression
template <unsigned int dim>
void FIELD_add_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field<double, 1>("u", mesh);
    u.fill(1.0);
    auto v = make_field<double, 1>("v", mesh);
    v.fill(1.0);
    auto w = make_field<double, 1>("w", mesh);
    w.fill(0.0);
    for (auto _ : state)
    {
        w = u + v;
        benchmark::DoNotOptimize(w[0]);
    }
}

BENCHMARK_TEMPLATE(MESH_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(MESH_uniform, 2)->DenseRange(1, 14);
;
BENCHMARK_TEMPLATE(MESH_uniform, 3)->DenseRange(1, 9);
;

BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 2)->DenseRange(1, 12);
;
BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 3)->DenseRange(1, 7);
;

BENCHMARK_TEMPLATE(FIELD_fill_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(FIELD_fill_uniform, 2)->DenseRange(1, 12);
;
BENCHMARK_TEMPLATE(FIELD_fill_uniform, 3)->DenseRange(1, 7);
;

BENCHMARK_TEMPLATE(FIELD_for_each_cell_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(FIELD_for_each_cell_uniform, 2)->DenseRange(1, 12);
;
BENCHMARK_TEMPLATE(FIELD_for_each_cell_uniform, 3)->DenseRange(1, 7);
;

BENCHMARK_TEMPLATE(FIELD_equal_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(FIELD_equal_uniform, 2)->DenseRange(1, 12);
;
BENCHMARK_TEMPLATE(FIELD_equal_uniform, 3)->DenseRange(1, 7);
;

BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 2)->DenseRange(1, 12);
;
BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 3)->DenseRange(1, 7);
;

BENCHMARK_TEMPLATE(FIELD_add_scalar_for_each_cell_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(FIELD_add_scalar_for_each_cell_uniform, 2)->DenseRange(1, 12);
;
BENCHMARK_TEMPLATE(FIELD_add_scalar_for_each_cell_uniform, 3)->DenseRange(1, 7);
;

BENCHMARK_TEMPLATE(FIELD_add_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(FIELD_add_uniform, 2)->DenseRange(1, 12);
;
BENCHMARK_TEMPLATE(FIELD_add_uniform, 3)->DenseRange(1, 7);
;

BENCHMARK_TEMPLATE(FIELD_add_for_each_cell_uniform, 1)->DenseRange(1, 16);
;
BENCHMARK_TEMPLATE(FIELD_add_for_each_cell_uniform, 2)->DenseRange(1, 12);
;
BENCHMARK_TEMPLATE(FIELD_add_for_each_cell_uniform, 3)->DenseRange(1, 7);
;
