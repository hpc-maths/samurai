// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Cost of samurai::find() (coordinate -> cell lookup) on a randomly-refined
// CellArray, in 1D/2D/3D, as the number of lookups per timed iteration grows
// (state.range(0)).
//
// find() is the primitive behind every coordinate-based query (boundary
// extrapolation, point-location tests, restart consistency checks); this
// isolates its cost from the set-algebra machinery that surrounds it
// elsewhere in the library. The mesh (generated once per fixture, not
// rebuilt per iteration) is refined by an unstructured coin flip per
// interval, not by MR adaptation, so its topology does not correspond to
// any physical solution field.

#include <array>
#include <random>

#include <benchmark/benchmark.h>

#include <xtensor/containers/xfixed.hpp>
#include <xtensor/generators/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/static_algorithm.hpp>

namespace
{
    // Fixed-seed uniform integer draw, reproducible across runs.
    template <class T>
    T randint(T lo, T hi)
    {
        static std::mt19937 gen{42};
        std::uniform_int_distribution<T> dist(lo, hi);
        return dist(gen);
    }
}

template <std::size_t dim>
auto generate_mesh(int bound, std::size_t start_level, std::size_t max_level)
{
    samurai::Box<int, dim> box({-bound << start_level, -bound << start_level, -bound << start_level},
                               {bound << start_level, bound << start_level, bound << start_level});
    samurai::CellArray<dim> ca;

    ca[start_level] = {start_level, box};

    for (std::size_t ite = 0; ite < max_level - start_level; ++ite)
    {
        samurai::CellList<dim> cl;

        samurai::for_each_interval(ca,
                                   [&](std::size_t level, const auto& interval, const auto& index)
                                   {
                                       auto choice = xt::random::choice(xt::xtensor_fixed<bool, xt::xshape<2>>{true, false}, interval.size());
                                       for (int i = interval.start, ic = 0; i < interval.end; ++i, ++ic)
                                       {
                                           if (choice[ic])
                                           {
                                               samurai::static_nested_loop<dim - 1, 0, 2>(
                                                   [&](auto stencil)
                                                   {
                                                       auto new_index = 2 * index + stencil;
                                                       cl[level + 1][new_index].add_interval({2 * i, 2 * i + 2});
                                                   });
                                           }
                                           else
                                           {
                                               cl[level][index].add_point(i);
                                           }
                                       }
                                   });

        ca = {cl, true};
    }

    return ca;
}

template <std::size_t dim_, int bound>
class MyFixture : public ::benchmark::Fixture
{
  public:

    static constexpr std::size_t dim       = dim_;
    static constexpr std::size_t min_level = 1;
    static constexpr std::size_t max_level = 10;

    MyFixture()
    {
        mesh = generate_mesh<dim_>(bound, min_level, max_level);
    }

    void bench(benchmark::State& state)
    {
        std::size_t found = 0;
        for (auto _ : state)
        {
            for (std::size_t s = 0; s < state.range(0); ++s)
            {
                auto level = randint(min_level, max_level);
                xt::xtensor_fixed<int, xt::xshape<dim>> coord;
                for (auto& c : coord)
                {
                    c = randint(-bound << level, (bound << level) - 1);
                }
                auto out = samurai::find(mesh[level], coord);
                if (out != -1)
                {
                    found++;
                }
            }
        }
        state.counters["nb cells"] = mesh.nb_cells();
        state.counters["found"]    = static_cast<double>(found) / state.iterations();
    }

    samurai::CellArray<dim_> mesh;
};

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_1D, 1, 1000)

(benchmark::State& state)
{
    bench(state);
}

BENCHMARK_REGISTER_F(MyFixture, Search_1D)->DenseRange(1, 10, 1);

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_2D, 2, 10)

(benchmark::State& state)
{
    bench(state);
}

BENCHMARK_REGISTER_F(MyFixture, Search_2D)->DenseRange(1, 10, 1);

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_3D, 3, 1)(benchmark::State& state)
{
    bench(state);
}

BENCHMARK_REGISTER_F(MyFixture, Search_3D)->DenseRange(1, 10, 1);
