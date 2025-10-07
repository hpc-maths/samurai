
#include <benchmark/benchmark.h>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/uniform_mesh.hpp>

// Observation
// - divide is coslty --> verify where and why we use it

// Mesure : Création d'un intervalle
void INTERVAL_default(benchmark::State& state)
{
    for (auto _ : state)
    {
        auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
        benchmark::DoNotOptimize(interval);
    }
}

// Mesure : Récupération de la taille d'un intervalle
void INTERVAL_size(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        auto size = interval.size();
        benchmark::DoNotOptimize(size);
    }
}

// Mesure : Test de validité d'un intervalle
void INTERVAL_is_valid(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        auto valid = interval.is_valid();
        benchmark::DoNotOptimize(valid);
    }
}

// Mesure : Test de parité d'un intervalle
void INTERVAL_even_elements(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        auto even = interval.even_elements();
        benchmark::DoNotOptimize(even);
    }
}

// Mesure : Test de parité d'un intervalle
void INTERVAL_odd_elements(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        auto odd = interval.odd_elements();
        benchmark::DoNotOptimize(odd);
    }
}

// Mesure : Multiplication d'un intervalle
void INTERVAL_multiply(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        interval *= 2;
        benchmark::DoNotOptimize(interval);
    }
}

// Mesure : divsion d'un intervalle
void INTERVAL_divide(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        interval /= 2;
        benchmark::DoNotOptimize(interval);
    }
}

// Here not a redundant test : to be sure that the condition "if start == end" is false.
void INTERVAL_multiply_divide(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        interval *= 2;
        interval /= 2;
        benchmark::DoNotOptimize(interval);
    }
}

// Mesure : Addition sur un intervalle
void INTERVAL_add(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        interval += 2;
        benchmark::DoNotOptimize(interval);
    }
}

// Mesure : Soustraction sur un intervalle
void INTERVAL_sub(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        interval -= 2;
        benchmark::DoNotOptimize(interval);
    }
}

// Mesure : Shift sur un intervalle
void INTERVAL_shift_increase(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        interval >>= 2;
        benchmark::DoNotOptimize(interval);
    }
}

// Mesure : Shift sur un intervalle
void INTERVAL_shift_decrease(benchmark::State& state)
{
    auto interval = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        interval <<= 2;
        benchmark::DoNotOptimize(interval);
    }
}

// Mesure : Test d'egalité d'un intervalle
void INTERVAL_equal(benchmark::State& state)
{
    auto interval1 = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    auto interval2 = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        auto is_equal = interval1 == interval2;
        benchmark::DoNotOptimize(is_equal);
    }
}

// Mesure : Test d'inégalité sur un intervalle
void INTERVAL_inequal(benchmark::State& state)
{
    auto interval1 = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    auto interval2 = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        auto is_inequal = interval1 != interval2;
        benchmark::DoNotOptimize(is_inequal);
    }
}

// Mesure : Comparaison sur un intervalle
void INTERVAL_less(benchmark::State& state)
{
    auto interval1 = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    auto interval2 = samurai::Interval<int64_t, int64_t>(0, 1, 0);
    for (auto _ : state)
    {
        auto is_less = interval1 < interval2;
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
