#include <iostream>

#include <benchmark/benchmark.h>
#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/subset/node_op.hpp>
#include <samurai/subset/subset_op.hpp>

template <std::size_t dim, class S>
inline auto init_sets_1(S& set1, S& set2, S& set3)
{
    std::size_t level = 8;
    samurai::Box<int, dim> box1({0, 0}, {1 << level, 1 << level});
    samurai::Box<int, dim> box2({0, 0}, {2, 2});
    samurai::Box<int, dim> box3({1, 1}, {2, 2});
    set1 = {level, box1};
    set2 = {level, box2};
    set3 = {level, box3};
}

template <std::size_t dim, class S>
inline auto init_sets_2(S& set1, S& set2, S& set3)
{
    std::size_t level = 8;
    samurai::Box<int, dim> box1({0, 0}, {1 << level, 1 << level});
    samurai::Box<int, dim> box2({0, 0}, {2, 2});
    samurai::Box<int, dim> box3({3, 3}, {4, 4});
    set1 = {level, box1};
    set2 = {level, box2};

    samurai::LevelCellList<dim> lcl(level + 1);
    lcl[{2}].add_interval({1, 2});
    lcl[{3}].add_interval({3, 4});
    set3 = {lcl};
}

static void BM_SetCreation(benchmark::State& state)
{
    constexpr std::size_t dim = 2;
    samurai::LevelCellArray<dim> set1, set2, set3;
    init_sets_1<dim>(set1, set2, set3);
    for (auto _ : state)
    {
        auto subset = samurai::intersection(samurai::intersection(set1, set2), set3);
    }
}

static void BM_SetOP(benchmark::State& state)
{
    constexpr std::size_t dim = 2;
    samurai::LevelCellArray<dim> set1, set2, set3;
    init_sets_1<dim>(set1, set2, set3);
    for (auto _ : state)
    {
        auto subset = samurai::intersection(samurai::intersection(set1, set2), set3);
        subset(
            [&](auto& interval, auto&)
            {
                interval *= 2;
            });
    }
}

static void BM_SetCreationWithOn(benchmark::State& state)
{
    constexpr std::size_t dim = 2;
    samurai::LevelCellArray<dim> set1, set2, set3;
    init_sets_1<dim>(set1, set2, set3);
    for (auto _ : state)
    {
        auto subset = samurai::intersection(samurai::intersection(set1, set2), set3).on(8);
    }
}

static void BM_SetOPWithOn(benchmark::State& state)
{
    constexpr std::size_t dim = 2;
    samurai::LevelCellArray<dim> set1, set2, set3;
    init_sets_1<dim>(set1, set2, set3);
    for (auto _ : state)
    {
        auto subset = samurai::intersection(samurai::intersection(set1, set2), set3).on(8);
        subset(
            [&](auto& interval, auto&)
            {
                interval *= 2;
            });
    }
}

static void BM_SetOPWithOn2(benchmark::State& state)
{
    constexpr std::size_t dim = 2;
    samurai::LevelCellArray<dim> set1, set2, set3;
    init_sets_1<dim>(set1, set2, set3);
    xt::xtensor_fixed<int, xt::xshape<dim>> stencil = {1, 0};

    for (auto _ : state)
    {
        auto subset = samurai::intersection(samurai::intersection(set1, samurai::translate(set2, stencil)), samurai::translate(set3, stencil))
                          .on(15);
        subset(
            [&](auto& interval, auto&)
            {
                interval *= 2;
            });
    }
}

static void BM_BigDomain(benchmark::State& state)
{
    constexpr std::size_t dim = 2;
    std::size_t level         = 12;
    samurai::Box<int, dim> box1({0, 0}, {1 << level, 1 << level});
    samurai::Box<int, dim> box2({1, 1}, {(1 << (level - 1)) - 1, (1 << (level - 1)) - 1});

    samurai::LevelCellArray<dim> set1{level, box1};
    samurai::LevelCellArray<dim> set2{level - 1, box2};

    std::size_t length = 0;
    for (auto _ : state)
    {
        length      = 0;
        auto subset = samurai::intersection(set1, set2).on(level - 2);
        subset(
            [&](const auto& interval, auto&)
            {
                length += interval.size();
            });
    }
}

BENCHMARK(BM_SetCreation);
BENCHMARK(BM_SetOP);
BENCHMARK(BM_SetCreationWithOn);
BENCHMARK(BM_SetOPWithOn);
BENCHMARK(BM_SetOPWithOn2);
BENCHMARK(BM_BigDomain);
