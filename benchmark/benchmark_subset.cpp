
#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/uniform_mesh.hpp>

// Observation : Cela prend environ 10ns par intervalle
// si on compare 2 intervalles de taillen, cela prendra environ 2n * 10ns

///////////////////////////////////////////////////////////////////
// utils

template <unsigned int dim>
auto cell_list_with_n_intervals(int64_t size)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < size; i++)
    {
        int index = static_cast<int>(i);
        cl[0][{}].add_interval({2 * index, 2 * index + 1});
    }
    return cl;
}

template <unsigned int dim>
auto cell_array_with_n_intervals(int64_t size)
{
    auto cl = cell_list_with_n_intervals<dim>(size);
    samurai::CellArray<dim> ca(cl);
    return ca;
}

//////////////////////////////////////////////////////////////////

/**
template <unsigned int dim, unsigned int delta_level >
void SUBSET_difference_same_interval_different_level(benchmark::State& state){
        samurai::CellList<dim> cl1, cl2 ;
        for (int64_t i = 0 ; i < state.range(0); i++){
                int index = static_cast<int>(i) ;
                cl1[0][{}].add_interval({2*index, 2*index+1});
                cl2[delta_level][{}].add_interval({pow(2, delta_level+1)*index, pow(2, delta_level+1)*index + pow(2, delta_level)});
        }
        samurai::CellArray<dim> ca1(cl1);
        samurai::CellArray<dim> ca2(cl2);
        for (auto _ : state){
        if constexpr(delta_level != 0){
                    auto subset = samurai::difference(ca1[0], samurai::projection(ca2[delta_level], 0));
                    subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
            benchmark::DoNotOptimize(subset);
        }
        else if constexpr (delta_level == 0){
                        auto subset = samurai::difference(ca1[0], ca2[0]);
                    subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
            benchmark::DoNotOptimize(subset);
        }
        }
}
**/
/**
template <unsigned int dim, unsigned int delta_level >
void SUBSET_difference_different_interval_different_level(benchmark::State& state){
        samurai::CellList<dim> cl1, cl2 ;
        for (int64_t i = 0 ; i < state.range(0); i++){
                int index = static_cast<int>(i) ;
                cl1[0][{}].add_interval({2*index, 2*index+1});
                cl2[delta_level][{}].add_interval({pow(2, delta_level+1)*index + pow(2, delta_level), pow(2, delta_level+2)*index});
        }
        samurai::CellArray<dim> ca1(cl1);
        samurai::CellArray<dim> ca2(cl2);
        for (auto _ : state){
                if constexpr(delta_level != 0){
                        auto subset = samurai::difference(ca1[0], samurai::projection(ca2[delta_level], 0));
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
                else if constexpr (delta_level == 0){
                        auto subset = samurai::difference(ca1[0], ca2[0]);
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
        }
}
**/

/**
template <unsigned int dim, unsigned int delta_level >
void SUBSET_difference_n1_interval_different_level(benchmark::State& state){
        samurai::CellList<dim> cl1, cl2 ;
        for (int64_t i = 0 ; i < state.range(0); i++){
                int index = static_cast<int>(i) ;
                cl1[0][{}].add_interval({2*index, 2*index+1});
        }
        cl2[delta_level][{}].add_interval({static_cast<int>(0), static_cast<int>(pow(2, delta_level))});

        samurai::CellArray<dim> ca1(cl1);
        samurai::CellArray<dim> ca2(cl2);
        for (auto _ : state){
                if constexpr(delta_level != 0){
                        auto subset = samurai::difference(ca1[0], samurai::projection(ca2[delta_level], 0));
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
                else if constexpr (delta_level == 0){
                        auto subset = samurai::difference(ca1[0], ca2[0]);
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
        }
}
**/

/**
template <unsigned int dim, unsigned int delta_level >
void SUBSET_double_difference_same_interval(benchmark::State& state){
        samurai::CellList<dim> cl1, cl2 ;
        for (int64_t i = 0 ; i < state.range(0); i++){
                int index = static_cast<int>(i) ;
                cl1[0][{}].add_interval({2*index, 2*index+1});
                cl2[delta_level][{}].add_interval({pow(2, delta_level+1)*index, pow(2, delta_level+1)*index + pow(2, delta_level)});
        }
        samurai::CellArray<dim> ca1(cl1);
        samurai::CellArray<dim> ca2(cl2);
        for (auto _ : state){
                if constexpr(delta_level != 0){
                        auto subset = samurai::difference(samurai::difference(ca1[0], samurai::projection(ca2[delta_level], 0)), ca1[0]);
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
                else if constexpr (delta_level == 0){
                        auto subset = samurai::difference(samurai::difference(ca1[0], ca2[0]), ca1[0]);
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
        }
}
**/

/**
// why for <2-3, 1-2-10> it is that fast ? error i think
template <unsigned int dim, unsigned int delta_level >
void SUBSET_intersection_same_interval_different_level(benchmark::State& state){
        samurai::CellList<dim> cl1, cl2 ;
        for (int64_t i = 0 ; i < state.range(0); i++){
                int index = static_cast<int>(i) ;
                cl1[0][{}].add_interval({2*index, 2*index+1});
                cl2[delta_level][{}].add_interval({pow(2, delta_level+1)*index, pow(2, delta_level+1)*index + pow(2, delta_level)});

        }
        samurai::CellArray<dim> ca1(cl1);
        samurai::CellArray<dim> ca2(cl2);
        for (auto _ : state){
                if constexpr(delta_level != 0){
                        auto subset = samurai::intersection(ca1[0], samurai::projection(ca2[delta_level], 0));
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
                else if constexpr (delta_level == 0){
                        auto subset = samurai::intersection(ca1[0], ca2[0]);
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
        }
}
**/
/**
template <unsigned int dim, unsigned int delta_level >
void SUBSET_union_same_interval_different_level(benchmark::State& state){
        samurai::CellList<dim> cl1, cl2 ;
        for (int64_t i = 0 ; i < state.range(0); i++){
                int index = static_cast<int>(i) ;
                cl1[0][{}].add_interval({2*index, 2*index+1});
                cl2[delta_level][{}].add_interval({pow(2, delta_level+1)*index, pow(2, delta_level+1)*index + pow(2, delta_level)});

        }
        samurai::CellArray<dim> ca1(cl1);
        samurai::CellArray<dim> ca2(cl2);
        for (auto _ : state){
                if constexpr(delta_level != 0){
                        auto subset = samurai::union_(ca1[0], samurai::projection(ca2[delta_level], 0));
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
                else if constexpr (delta_level == 0){
                        auto subset = samurai::union_(ca1[0], ca2[0]);
                        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                        benchmark::DoNotOptimize(subset);
                }
        }
}
**/

template <unsigned int dim>
void SUBSET_translate(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        cl[0][{}].add_interval({2 * index, 2 * index + 1});
    }
    samurai::CellArray<dim> ca(cl);
    xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
    if constexpr (dim == 1)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<1>>({1});
    }
    else if constexpr (dim == 2)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
    }
    else if constexpr (dim == 3)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<3>>({1, 1, 1});
    }
    for (auto _ : state)
    {
        auto subset = samurai::translate(ca[0], stencil);
        subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
        benchmark::DoNotOptimize(subset);
    }
}

/**
template <unsigned int dim>
void SUBSET_expand(benchmark::State& state){
        samurai::CellList<dim> cl ;
        for (int64_t i = 0 ; i < state.range(0); i++){
                int index = static_cast<int>(i) ;
                cl[0][{}].add_interval({2*index, 2*index+1});
        }
        samurai::CellArray<dim> ca(cl);
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
        if constexpr (dim == 1)
                stencil = xt::xtensor_fixed<int, xt::xshape<1>>({1}) ;
        else if constexpr (dim == 2)
                stencil = xt::xtensor_fixed<int, xt::xshape<2>>({1,1}) ;
        else if constexpr (dim == 3)
                stencil = xt::xtensor_fixed<int, xt::xshape<3>>({1,1,1}) ;
        for (auto _ : state){
                auto subset = samurai::expand(ca[0]);
                subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                benchmark::DoNotOptimize(subset);
        }
}
**/

/**
template <unsigned int dim>
void SUBSET_contraction(benchmark::State& state){
        samurai::CellList<dim> cl ;
        for (int64_t i = 0 ; i < state.range(0); i++){
                int index = static_cast<int>(i) ;
                cl[0][{}].add_interval({2*index, 2*index+1});
        }
        samurai::CellArray<dim> ca(cl);
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
        if constexpr (dim == 1)
                stencil = xt::xtensor_fixed<int, xt::xshape<1>>({1}) ;
        else if constexpr (dim == 2)
                stencil = xt::xtensor_fixed<int, xt::xshape<2>>({1,1}) ;
        else if constexpr (dim == 3)
                stencil = xt::xtensor_fixed<int, xt::xshape<3>>({1,1,1}) ;
        for (auto _ : state){
                auto subset = samurai::contraction(ca[0]);
                subset([](const auto&, const auto&) {}); // evaluation avec lambda vide
                benchmark::DoNotOptimize(subset);
        }
}
**/
/**
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,1, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,2, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,3, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,1, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,2, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,3, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,1, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,2, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,3, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,1, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,2, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_same_interval_different_level,3, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
**/
/**
BENCHMARK_TEMPLATE(SUBSET_difference_n1_interval_different_level,1, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_n1_interval_different_level,2, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_difference_n1_interval_different_level,3, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
**/

/**
BENCHMARK_TEMPLATE(SUBSET_double_difference_same_interval,1, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_double_difference_same_interval,2, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_double_difference_same_interval,3, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
**/

/**
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,1, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,2, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,3, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,1, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,2, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,3, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,1, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,2, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,3, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,1, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,2, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_intersection_same_interval_different_level,3, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
**/

/**
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,1, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,2, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,3, 0)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,1, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,2, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,3, 1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,1, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,2, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,3, 2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,1, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,2, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_union_same_interval_different_level,3, 10)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
**/

// BENCHMARK_TEMPLATE(SUBSET_translate,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
// BENCHMARK_TEMPLATE(SUBSET_translate,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
// BENCHMARK_TEMPLATE(SUBSET_translate,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

/**
BENCHMARK_TEMPLATE(SUBSET_expand,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_expand,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_expand,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
**/

// BENCHMARK_TEMPLATE(SUBSET_contraction,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
// BENCHMARK_TEMPLATE(SUBSET_contraction,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
// BENCHMARK_TEMPLATE(SUBSET_contraction,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
