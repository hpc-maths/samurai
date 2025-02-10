#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/uniform_mesh.hpp>
#include <samurai/amr/mesh.hpp>



template <unsigned int dim>
void MESH_default(benchmark::State& state){
	using Config = samurai::amr::Config<dim> ;
        for (auto _ : state){
		auto mesh = samurai::amr::Mesh<Config>() ; 
                benchmark::DoNotOptimize(mesh);		
        }
}

BENCHMARK_TEMPLATE(MESH_default, 1);
BENCHMARK_TEMPLATE(MESH_default, 2);
BENCHMARK_TEMPLATE(MESH_default, 3);




