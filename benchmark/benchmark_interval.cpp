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
/////////////////////////////////////////////////////////////


void INTERVAL_default(benchmark::State& state){
        for (auto _ : state){
                auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
                benchmark::DoNotOptimize(interval);
        }
}

void INTERVAL_size(benchmark::State& state){
	auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
	for (auto _ : state){
		auto size = interval.size() ; 
                benchmark::DoNotOptimize(size);
        }
}

void INTERVAL_is_valid(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                auto valid = interval.is_valid() ;
                benchmark::DoNotOptimize(valid);
        }
}


void INTERVAL_even_elements(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                auto even = interval.even_elements() ;
                benchmark::DoNotOptimize(even);
        }
}

void INTERVAL_odd_elements(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                auto odd = interval.odd_elements() ;
                benchmark::DoNotOptimize(odd);
        }
}

void INTERVAL_multiply(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                interval *= 2 ;
                benchmark::DoNotOptimize(interval);
        }
}

void INTERVAL_divide(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                interval /= 2 ;
                benchmark::DoNotOptimize(interval);
        }
}

void INTERVAL_add(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                interval += 2 ;
                benchmark::DoNotOptimize(interval);
        }
}

void INTERVAL_sub(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                interval -= 2 ;
                benchmark::DoNotOptimize(interval);
        }
}


void INTERVAL_shift_increase(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                interval >>= 2 ;
                benchmark::DoNotOptimize(interval);
        }
}

void INTERVAL_shift_decrease(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                interval <<= 2 ;
                benchmark::DoNotOptimize(interval);
        }
}


void INTERVAL_equal(benchmark::State& state){
        auto interval1 = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        auto interval2 = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;	
        for (auto _ : state){
                auto is_equal = interval1 == interval2 ;
                benchmark::DoNotOptimize(is_equal);
        }
}

void INTERVAL_inequal(benchmark::State& state){
        auto interval1 = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        auto interval2 = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                auto is_inequal = interval1 != interval2 ;
                benchmark::DoNotOptimize(is_inequal);
        }
}

void INTERVAL_less(benchmark::State& state){
        auto interval1 = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        auto interval2 = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                auto is_less = interval1 < interval2 ;
                benchmark::DoNotOptimize(is_less);
        }
}



template <unsigned int dim>
void INTERVAL_cl_add_interval_end(benchmark::State& state){
	samurai::CellList<dim> cl ; 
        for (auto _ : state){
		for (int64_t i = 0 ; i < state.range(0); i++){
			int index = static_cast<int>(i) ; 
	                cl[0][{}].add_interval({index, index+1});
		}
        }
}


template <unsigned int dim>
void INTERVAL_cl_add_interval_begin(benchmark::State& state){
        samurai::CellList<dim> cl ;
        for (auto _ : state){
                for (int64_t i = state.range(0); i > 0; i--){
                        int index = static_cast<int>(i) ;			
                        cl[0][{}].add_interval({index, index+1});
                }
        }
}

template <unsigned int dim>
void INTERVAL_cl_add_interval_same(benchmark::State& state){
        samurai::CellList<dim> cl ;
        for (auto _ : state){
                for (int64_t i = state.range(0); i > 0; i--){
                        cl[0][{}].add_interval({0, 1});
                }
        }
}

template <unsigned int dim>
void INTERVAL_cl_ca_one(benchmark::State& state){
        samurai::CellList<dim> cl ;
        cl[0][{}].add_interval({0, 1});
        for (auto _ : state){
		samurai::CellArray<dim> ca(cl) ; 
		benchmark::DoNotOptimize(ca[0]);
        }
}


template <unsigned int dim>
void INTERVAL_cl_ca_multi(benchmark::State& state){
	auto cl = cell_list_with_n_intervals<dim>(state.range(0)) ;
        for (auto _ : state){
                samurai::CellArray<dim> ca(cl) ;
                benchmark::DoNotOptimize(ca[0]);
        }
}



BENCHMARK(INTERVAL_default);
BENCHMARK(INTERVAL_size);
BENCHMARK(INTERVAL_is_valid);
BENCHMARK(INTERVAL_even_elements);
BENCHMARK(INTERVAL_odd_elements);
BENCHMARK(INTERVAL_multiply);
BENCHMARK(INTERVAL_divide);
BENCHMARK(INTERVAL_add);
BENCHMARK(INTERVAL_sub);

BENCHMARK(INTERVAL_shift_increase);
BENCHMARK(INTERVAL_shift_decrease);

BENCHMARK(INTERVAL_equal);
BENCHMARK(INTERVAL_inequal);
BENCHMARK(INTERVAL_less);



BENCHMARK_TEMPLATE(INTERVAL_cl_add_interval_end,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(INTERVAL_cl_add_interval_end,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(INTERVAL_cl_add_interval_end,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(INTERVAL_cl_add_interval_begin,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(INTERVAL_cl_add_interval_same,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(INTERVAL_cl_ca_one,1); 
BENCHMARK_TEMPLATE(INTERVAL_cl_ca_one,2);
BENCHMARK_TEMPLATE(INTERVAL_cl_ca_one,3);


BENCHMARK_TEMPLATE(INTERVAL_cl_ca_multi,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(INTERVAL_cl_ca_multi,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(INTERVAL_cl_ca_multi,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);


