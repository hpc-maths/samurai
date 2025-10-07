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
#include <samurai/stencil.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/uniform_mesh.hpp>

//////////////////////////////////////////////////////////////////////////////
// BENCHMARKS - CRÉATION DE STENCILS FONDAMENTAUX
//////////////////////////////////////////////////////////////////////////////

// Benchmark de création des stencils star (le plus fondamental)
template <std::size_t dim, std::size_t neighbourhood_width>
void STENCIL_star_creation(benchmark::State& state)
{
    for (auto _ : state)
    {
        auto stencil = samurai::star_stencil<dim, neighbourhood_width>();
        benchmark::DoNotOptimize(stencil);
    }
}

//////////////////////////////////////////////////////////////////////////////
// REGISTRATIONS DES BENCHMARKS
//////////////////////////////////////////////////////////////////////////////

// Benchmarks Star stencil - configurations essentielles
BENCHMARK_TEMPLATE(STENCIL_star_creation, 2, 1); // 2D, largeur 1 (le plus courant)
BENCHMARK_TEMPLATE(STENCIL_star_creation, 3, 1); // 3D, largeur 1 (le plus courant)
BENCHMARK_TEMPLATE(STENCIL_star_creation, 2, 2); // 2D, largeur 2
BENCHMARK_TEMPLATE(STENCIL_star_creation, 3, 2); // 3D, largeur 2
