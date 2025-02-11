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
#include <samurai/stencil.hpp>





/**
template <std::size_t dim, std::size_t stencil_size>
void STENCIL_find_stencil_origin(benchmark::State& state){
	using Stencil = xt::xtensor_fixed<int, xt::xshape<stencil_size, dim>>;
//tencil stencil = xt::zeros<int>({stencil_size, dim});
	Stencil stencil = star_stencil() ; 
        for (auto _ : state){
		auto origin = samurai::find_stencil_origin(stencil) ; 
		benchmark::DoNotOptimize(origin) ; 
        }
}




BENCHMARK_TEMPLATE(STENCIL_find_stencil_origin, 1, 1);
BENCHMARK_TEMPLATE(STENCIL_find_stencil_origin, 2, 1);
BENCHMARK_TEMPLATE(STENCIL_find_stencil_origin, 3, 1);

**/
