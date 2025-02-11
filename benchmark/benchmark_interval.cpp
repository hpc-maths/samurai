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

// Here not a redundant test : to be sure that the condition "if start == end" is false. 
void INTERVAL_multiply_divide(benchmark::State& state){
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0) ;
        for (auto _ : state){
                interval *= 2 ;
		interval /=2 ;
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



BENCHMARK(INTERVAL_default);
BENCHMARK(INTERVAL_size);
BENCHMARK(INTERVAL_is_valid);
BENCHMARK(INTERVAL_even_elements);
BENCHMARK(INTERVAL_odd_elements);
BENCHMARK(INTERVAL_multiply);
BENCHMARK(INTERVAL_divide);
BENCHMARK(INTERVAL_multiply_divide);

BENCHMARK(INTERVAL_add);
BENCHMARK(INTERVAL_sub);

BENCHMARK(INTERVAL_shift_increase);
BENCHMARK(INTERVAL_shift_decrease);

BENCHMARK(INTERVAL_equal);
BENCHMARK(INTERVAL_inequal);
BENCHMARK(INTERVAL_less);




