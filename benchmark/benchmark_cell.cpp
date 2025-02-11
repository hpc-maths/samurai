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

template <std::size_t dim>
auto make_cell(){
        using value_t = samurai::default_config::value_t ;
        using point_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;
        point_t indice ;
        point_t begin ;

	if constexpr (dim == 1){
		indice = {1} ; 
		begin = {0} ; 
	}
        if constexpr (dim == 2){
                indice = {1,1} ;
                begin = {0,0} ;
        }	
        if constexpr (dim == 3){
                indice = {1,1,1} ;
                begin = {0,0,0} ;
        }

	auto indices = xt::xtensor_fixed<int, xt::xshape<dim>>(indice) ;
        double scaling_factor = 1.0 ;
        auto c = samurai::Cell<dim, samurai::Interval<int>>(begin, scaling_factor, 1, indices, 0);
	return c ; 
}

template <unsigned int dim>
void CELL_default(benchmark::State& state){
        for (auto _ : state){
                auto c = samurai::Cell<dim, samurai::Interval<int>>();
                benchmark::DoNotOptimize(c);
        }
}


template <unsigned int dim>
void CELL_init(benchmark::State& state){
	auto indices = xt::xtensor_fixed<int, xt::xshape<2>>({1,1}) ;
	double scaling_factor = 1.0 ; 	
        for (auto _ : state){
		auto c = samurai::Cell<2, samurai::Interval<int>>({0,0}, scaling_factor, 1, indices, 0);
		benchmark::DoNotOptimize(c); 
	}
}

template <unsigned int dim>
void CELL_center(benchmark::State& state){
	auto c = make_cell<dim>();
        for (auto _ : state){
		auto center = c.center() ; 
                benchmark::DoNotOptimize(center);
        }
}

template <unsigned int dim>
void CELL_face_center(benchmark::State& state){
        auto c = make_cell<dim>();
        for (auto _ : state){
                auto center = c.face_center() ;
                benchmark::DoNotOptimize(center);
        }
}


template <unsigned int dim>
void CELL_corner(benchmark::State& state){
        auto c = make_cell<dim>();
        for (auto _ : state){
                auto center = c.corner() ;
                benchmark::DoNotOptimize(center);
        }
}

template <unsigned int dim>
void CELL_equal(benchmark::State& state){
        auto c1 = make_cell<dim>();
        auto c2 = make_cell<dim>();
        for (auto _ : state){
		auto is_equal = c1 == c2 ; 
                benchmark::DoNotOptimize(is_equal);
        }
}

template <unsigned int dim>
void CELL_different(benchmark::State& state){
        auto c1 = make_cell<dim>();
        auto c2 = make_cell<dim>();
        for (auto _ : state){
                auto is_equal = c1 != c2 ;
                benchmark::DoNotOptimize(is_equal);
        }
}



BENCHMARK_TEMPLATE(CELL_default,1);
BENCHMARK_TEMPLATE(CELL_default,2);
BENCHMARK_TEMPLATE(CELL_default,3);
BENCHMARK_TEMPLATE(CELL_default,4);


BENCHMARK_TEMPLATE(CELL_init,2);

BENCHMARK_TEMPLATE(CELL_center,1);
BENCHMARK_TEMPLATE(CELL_center,2);
BENCHMARK_TEMPLATE(CELL_center,3);

BENCHMARK_TEMPLATE(CELL_corner,1);
BENCHMARK_TEMPLATE(CELL_corner,2);
BENCHMARK_TEMPLATE(CELL_corner,3);

BENCHMARK_TEMPLATE(CELL_equal,1);
BENCHMARK_TEMPLATE(CELL_equal,2);
BENCHMARK_TEMPLATE(CELL_equal,3);

BENCHMARK_TEMPLATE(CELL_different,1);
BENCHMARK_TEMPLATE(CELL_different,2);
BENCHMARK_TEMPLATE(CELL_different,3);

