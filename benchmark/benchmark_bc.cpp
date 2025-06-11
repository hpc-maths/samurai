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

////////////////////////////////////////////////////////////////////////////////////
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

// Mesure : Création d'une condition limite sur un maillage uniforme
template <unsigned int dim, unsigned int n_comp, typename BCType, unsigned int order>
void BC_homogeneous(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config = samurai::UniformConfig<dim, order>;  // Utilisation de l'ordre comme paramètre
    auto mesh = samurai::UniformMesh<Config>(box, state.range(0));
    auto u = make_field<double, n_comp>("u", mesh);
    
    u.fill(1.0);
    
    for (auto _ : state)
    {
        samurai::make_bc<BCType>(u);
    }
}

// Test BC sur une direction spécifique
template <unsigned int dim, unsigned int n_comp, typename BCType, unsigned int order>
void BC_directional(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config = samurai::UniformConfig<dim, order>;
    auto mesh = samurai::UniformMesh<Config>(box, state.range(0));
    auto u = make_field<double, n_comp>("u", mesh);
    
    u.fill(1.0);
    
    for (auto _ : state)
    {
        if constexpr (dim == 1)
        {
            const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
            samurai::make_bc<BCType>(u)->on(left);
        }
        else if constexpr (dim == 2)
        {
            const xt::xtensor_fixed<int, xt::xshape<2>> left{-1, 0};
            samurai::make_bc<BCType>(u)->on(left);
        }
        else if constexpr (dim == 3)
        {
            const xt::xtensor_fixed<int, xt::xshape<3>> left{-1, 0, 0};
            samurai::make_bc<BCType>(u)->on(left);
        }
    }
}

// Tests Dirichlet
BENCHMARK_TEMPLATE(BC_homogeneous, 1, 1, samurai::Dirichlet<1>, 1)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 2, 1, samurai::Dirichlet<1>, 1)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 3, 1, samurai::Dirichlet<1>, 1)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 1, 100, samurai::Dirichlet<1>, 1)->DenseRange(1, 1);

// Tests Neumann
BENCHMARK_TEMPLATE(BC_homogeneous, 1, 1, samurai::Neumann<1>, 1)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 2, 1, samurai::Neumann<1>, 1)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 3, 1, samurai::Neumann<1>, 1)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 1, 100, samurai::Neumann<1>, 1)->DenseRange(1, 1);

// Tests Dirichlet ordre 3
BENCHMARK_TEMPLATE(BC_homogeneous, 1, 1, samurai::Dirichlet<2>, 3)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 2, 1, samurai::Dirichlet<2>, 3)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 3, 1, samurai::Dirichlet<2>, 3)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 1, 100, samurai::Dirichlet<2>, 3)->DenseRange(1, 1);

// BC directionnels
BENCHMARK_TEMPLATE(BC_directional, 1, 1, samurai::Dirichlet<1>, 1)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_directional, 2, 1, samurai::Dirichlet<1>, 1)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_directional, 3, 1, samurai::Dirichlet<1>, 1)->DenseRange(1, 1);




