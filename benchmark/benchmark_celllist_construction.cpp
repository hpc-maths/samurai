#include <benchmark/benchmark.h>
#include <experimental/random>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>


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
void CELLLIST_cl_add_interval_end(benchmark::State& state){
        samurai::CellList<dim> cl ;
        for (auto _ : state){
                for (int64_t i = 0 ; i < state.range(0); i++){
                        int index = static_cast<int>(i) ;
                        cl[0][{}].add_interval({index, index+1});
                }
        }
}

template <unsigned int dim>
void CELLLIST_cl_add_interval_begin(benchmark::State& state){
        samurai::CellList<dim> cl ;
        for (auto _ : state){
                for (int64_t i = state.range(0); i > 0; i--){
                        int index = static_cast<int>(i) ;
                        cl[0][{}].add_interval({index, index+1});
                }
        }
}

template <unsigned int dim>
void CELLLIST_cl_add_interval_same(benchmark::State& state){
        samurai::CellList<dim> cl ;
        for (auto _ : state){
                for (int64_t i = state.range(0); i > 0; i--){
                        cl[0][{}].add_interval({0, 1});
                }
        }
}




static void BM_CellListConstruction_2D(benchmark::State& state)
{
    constexpr std::size_t dim = 2;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;

    for (auto _ : state)
    {
        for (std::size_t s = 0; s < state.range(0); ++s)
        {
            auto level = std::experimental::randint(min_level, max_level);
            auto x     = std::experimental::randint(0, (100 << level) - 1);
            auto y     = std::experimental::randint(0, (100 << level) - 1);

            cl[level][{y}].add_point(x);
        }
    }
}

BENCHMARK(BM_CellListConstruction_2D)->Range(8, 8 << 18);

static void BM_CellListConstruction_3D(benchmark::State& state)
{
    constexpr std::size_t dim = 3;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;

    for (auto _ : state)
    {
        for (std::size_t s = 0; s < state.range(0); ++s)
        {
            auto level = std::experimental::randint(min_level, max_level);
            auto x     = std::experimental::randint(0, (100 << level) - 1);
            auto y     = std::experimental::randint(0, (100 << level) - 1);
            auto z     = std::experimental::randint(0, (100 << level) - 1);

            cl[level][{y, z}].add_point(x);
        }
    }
}

BENCHMARK(BM_CellListConstruction_3D)->Range(8, 8 << 18);

static void BM_CellList2CellArray_2D(benchmark::State& state)
{
    constexpr std::size_t dim = 2;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;
    samurai::CellArray<dim> ca;

    for (std::size_t s = 0; s < state.range(0); ++s)
    {
        auto level = std::experimental::randint(min_level, max_level);
        auto x     = std::experimental::randint(0, (100 << level) - 1);
        auto y     = std::experimental::randint(0, (100 << level) - 1);

        cl[level][{y}].add_point(x);
    }

    for (auto _ : state)
    {
        ca = {cl};
    }
}

BENCHMARK(BM_CellList2CellArray_2D)->Range(8, 8 << 18);

static void BM_CellList2CellArray_3D(benchmark::State& state)
{
    constexpr std::size_t dim = 3;

    std::size_t min_level = 1;
    std::size_t max_level = 12;

    samurai::CellList<dim> cl;
    samurai::CellArray<dim> ca;

    for (std::size_t s = 0; s < state.range(0); ++s)
    {
        auto level = std::experimental::randint(min_level, max_level);
        auto x     = std::experimental::randint(0, (100 << level) - 1);
        auto y     = std::experimental::randint(0, (100 << level) - 1);
        auto z     = std::experimental::randint(0, (100 << level) - 1);

        cl[level][{y, z}].add_point(x);
    }

    for (auto _ : state)
    {
        ca = {cl};
    }
}

BENCHMARK(BM_CellList2CellArray_3D)->Range(8, 8 << 18);



BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_end,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_end,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_end,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_begin,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(CELLLIST_cl_add_interval_same,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);


