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


template <unsigned int dim>
auto cell_list_with_n_intervals(int64_t size){
        samurai::CellList<dim> cl ;
        for (int64_t i = 0 ; i < size; i++){
                int index = static_cast<int>(i) ;
                cl[0][{}].add_interval({2*index, 2*index+1});
        }
	return cl ; 
}

template <unsigned int dim>
auto cell_array_with_n_intervals(int64_t size){
	auto cl = cell_list_with_n_intervals<dim>(size) ;  
	samurai::CellArray<dim> ca(cl) ;
	return ca;
}



template <unsigned int dim, unsigned int max_level>
void CELLARRAY_default(benchmark::State& state){
        for (auto _ : state){
                samurai::CellArray ca = samurai::CellArray<dim, samurai::default_config::interval_t, max_level>();
        }
}




BENCHMARK_TEMPLATE(CELLARRAY_default,1, 16);
BENCHMARK_TEMPLATE(CELLARRAY_default,2, 16);
BENCHMARK_TEMPLATE(CELLARRAY_default,3, 16);

