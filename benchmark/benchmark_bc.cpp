#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/bc.hpp>
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
    using Config                  = samurai::UniformConfig<dim, static_cast<int>(order)>; // ghost_width = order
    auto mesh                     = samurai::UniformMesh<Config>(box, static_cast<std::size_t>(state.range(0)));
    auto u                        = samurai::make_vector_field<double, n_comp>("u", mesh);

    u.fill(1.0);

    // Ajout des compteurs personnalisés
    state.counters["Dimension"]   = dim;
    state.counters["Composantes"] = n_comp;
    state.counters["Ordre"]       = order;
    state.counters["Type BC"]     = std::is_same_v<BCType, samurai::Dirichlet<order>> ? 0 : 1; // 0 pour Dirichlet, 1 pour Neumann

    auto& bc_container = u.get_bc();

    for (auto _ : state)
    {
        state.PauseTiming();
        bc_container.clear();
        state.ResumeTiming();
        samurai::make_bc<BCType>(u);
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
BENCHMARK_TEMPLATE(BC_homogeneous, 1, 1, samurai::Dirichlet<3>, 3)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 2, 1, samurai::Dirichlet<3>, 3)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 3, 1, samurai::Dirichlet<3>, 3)->DenseRange(1, 1);
BENCHMARK_TEMPLATE(BC_homogeneous, 1, 100, samurai::Dirichlet<3>, 3)->DenseRange(1, 1);
