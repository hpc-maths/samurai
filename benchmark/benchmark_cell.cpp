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

//////////////////////////////////////////////////////////////////
/// utils

template <std::size_t dim>
auto make_cell()
{
    using value_t = samurai::default_config::value_t;
    using point_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;

    // Initialisation générique des indices et du point de départ
    point_t indice = xt::ones<value_t>({dim});
    point_t begin  = xt::zeros<value_t>({dim});

    auto indices          = xt::xtensor_fixed<int, xt::xshape<dim>>(indice);
    double scaling_factor = 1.0;
    auto c                = samurai::Cell<dim, samurai::Interval<int>>(begin, scaling_factor, 1, indices, 0);
    return c;
}

// Mesure : Construction d'une cellule par défaut
template <unsigned int dim>
void CELL_default(benchmark::State& state)
{
    for (auto _ : state)
    {
        auto c = samurai::Cell<dim, samurai::Interval<int>>();
        benchmark::DoNotOptimize(c);
    }
}

// Mesure : Initialisation d'une cellule
template <unsigned int dim>
void CELL_init(benchmark::State& state)
{
    xt::xtensor_fixed<int, xt::xshape<dim>> indices;
    if constexpr (dim == 1)
    {
        indices = xt::xtensor_fixed<int, xt::xshape<1>>({1});
    }
    else if constexpr (dim == 2)
    {
        indices = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
    }
    else if constexpr (dim == 3)
    {
        indices = xt::xtensor_fixed<int, xt::xshape<3>>({1, 1, 1});
    }
    double scaling_factor = 1.0;
    for (auto _ : state)
    {
        if constexpr (dim == 1)
        {
            auto c = samurai::Cell<1, samurai::Interval<int>>({0}, scaling_factor, 1, indices, 0);
            benchmark::DoNotOptimize(c);
        }
        else if constexpr (dim == 2)
        {
            auto c = samurai::Cell<2, samurai::Interval<int>>({0, 0}, scaling_factor, 1, indices, 0);
            benchmark::DoNotOptimize(c);
        }
        else if constexpr (dim == 3)
        {
            auto c = samurai::Cell<3, samurai::Interval<int>>({0, 0, 0}, scaling_factor, 1, indices, 0);
            benchmark::DoNotOptimize(c);
        }
    }
}

// Mesure : Récupération du centre d'une cellule
template <unsigned int dim>
void CELL_center(benchmark::State& state)
{
    auto c = make_cell<dim>();
    for (auto _ : state)
    {
        auto center = c.center();
        benchmark::DoNotOptimize(center);
    }
}

// Mesure : Récupérayion du centre de la face d'une cellule
template <unsigned int dim>
void CELL_face_center(benchmark::State& state)
{
    auto c = make_cell<dim>();
    for (auto _ : state)
    {
        auto center = c.face_center();
        benchmark::DoNotOptimize(center);
    }
}

// Mesure : Récupération de l'angle d'une cellule
template <unsigned int dim>
void CELL_corner(benchmark::State& state)
{
    auto c = make_cell<dim>();
    for (auto _ : state)
    {
        auto center = c.corner();
        benchmark::DoNotOptimize(center);
    }
}

// Mesure : Test d'égalité entre deux cellules
template <unsigned int dim>
void CELL_equal(benchmark::State& state)
{
    auto c1 = make_cell<dim>();
    auto c2 = make_cell<dim>();
    for (auto _ : state)
    {
        auto is_equal = c1 == c2;
        benchmark::DoNotOptimize(is_equal);
    }
}

// Mesure : Test d'inégalité entre deux cellules
template <unsigned int dim>
void CELL_different(benchmark::State& state)
{
    auto c1 = make_cell<dim>();
    auto c2 = make_cell<dim>();
    for (auto _ : state)
    {
        auto is_equal = c1 != c2;
        benchmark::DoNotOptimize(is_equal);
    }
}

BENCHMARK_TEMPLATE(CELL_default, 1);
BENCHMARK_TEMPLATE(CELL_default, 2);
BENCHMARK_TEMPLATE(CELL_default, 3);
BENCHMARK_TEMPLATE(CELL_default, 4);

BENCHMARK_TEMPLATE(CELL_init, 1);
BENCHMARK_TEMPLATE(CELL_init, 2);
BENCHMARK_TEMPLATE(CELL_init, 3);

BENCHMARK_TEMPLATE(CELL_center, 1);
BENCHMARK_TEMPLATE(CELL_center, 2);
BENCHMARK_TEMPLATE(CELL_center, 3);

BENCHMARK_TEMPLATE(CELL_corner, 1);
BENCHMARK_TEMPLATE(CELL_corner, 2);
BENCHMARK_TEMPLATE(CELL_corner, 3);

BENCHMARK_TEMPLATE(CELL_equal, 1);
BENCHMARK_TEMPLATE(CELL_equal, 2);
BENCHMARK_TEMPLATE(CELL_equal, 3);

BENCHMARK_TEMPLATE(CELL_different, 1);
BENCHMARK_TEMPLATE(CELL_different, 2);
BENCHMARK_TEMPLATE(CELL_different, 3);
