// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Helpers shared by the set algebra benchmarks: the traversal of a set and its
// counters, and synthetic LevelCellArrays at one level, each shape chosen to
// stress one property of the traversal. Needs only the interval structures
// and the set algebra; the MR-adapted meshes are in mr_fixtures.hpp.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <xtensor/core/xoperation.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/views/xview.hpp>

#include <samurai/box.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/stencil.hpp>
#include <samurai/subset/node.hpp>

#include "set_algebra_backend.hpp"

namespace bench
{
    //////////////////////////////////////////////////////////////////////
    // Consuming a set
    //////////////////////////////////////////////////////////////////////

    struct Tally
    {
        std::size_t intervals = 0;
        std::size_t cells     = 0;

        template <class Interval, class Index>
        void operator()(const Interval& i, const Index&)
        {
            ++intervals;
            cells += static_cast<std::size_t>(i.size());
        }

        Tally& operator+=(const Tally& other)
        {
            intervals += other.intervals;
            cells += other.cells;
            return *this;
        }
    };

    // The callback consumes every interval: the compiler may otherwise drop
    // the traversal of a lazy expression. B is the backend of the set
    // (set_algebra_backend.hpp).
    template <class B = SamuraiStatic, class Set>
    Tally traverse(const Set& set)
    {
        Tally tally;
        Algebra<B>::traverse(set,
                             [&tally](const auto& i, const auto& index)
                             {
                                 tally(i, index);
                             });
        benchmark::DoNotOptimize(tally);
        return tally;
    }

    // The intervals of a LevelCellArray, with their level and (y, z): the
    // loop of samurai::for_each_interval (algorithm.hpp), which pulls the
    // mesh and the OpenMP loops.
    template <std::size_t dim, class TInterval, class F>
    void for_each_interval(const samurai::LevelCellArray<dim, TInterval>& lca, F&& f)
    {
        if (lca.empty())
        {
            return;
        }
        for (auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            f(lca.level(), *it, it.index());
        }
    }

    // `t/itv`, the time per emitted interval, is the figure to compare across
    // shapes, operators and dimensions.
    inline void report(benchmark::State& state, const Tally& tally)
    {
        state.counters["itv"]   = static_cast<double>(tally.intervals);
        state.counters["cells"] = static_cast<double>(tally.cells);
        state.counters["t/itv"] = benchmark::Counter(static_cast<double>(tally.intervals),
                                                     benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    }

    // The expression is rebuilt at every iteration, as the library does.
    template <class Make>
    void run(benchmark::State& state, Make&& make)
    {
        Tally tally;
        for (auto _ : state)
        {
            tally = traverse(make());
        }
        report(state, tally);
    }

    // `site` passes each set it evaluates to f(level, set), a set of the
    // backend B, which returns the number of intervals of the set: a site
    // that branches on whether a set is empty reads it there. Returns the
    // number of sets of one iteration.
    template <class B, class Site>
    std::size_t run_site(benchmark::State& state, Site&& site)
    {
        Tally tally;
        std::size_t nsets = 0;
        for (auto _ : state)
        {
            tally = {};
            nsets = 0;
            site(
                [&](std::size_t, const auto& set)
                {
                    const Tally t = traverse<B>(set);
                    tally += t;
                    ++nsets;
                    return t.intervals;
                });
        }
        report(state, tally);
        return nsets;
    }

    // `site` builds level cell arrays of the backend B and passes each one it
    // keeps to emit(level, lca). The time is that of building them; their
    // intervals and cells are counted after the timed loop, on one more run.
    // Returns the number of level cell arrays of one iteration.
    template <class B, class Site>
    std::size_t run_build_site(benchmark::State& state, Site&& site)
    {
        for (auto _ : state)
        {
            site(
                [](std::size_t, const auto& lca)
                {
                    const auto* p = &lca;
                    benchmark::DoNotOptimize(p);
                    benchmark::ClobberMemory();
                });
        }
        Tally tally;
        std::size_t nsets = 0;
        site(
            [&](std::size_t, const auto& lca)
            {
                tally += traverse<B>(lca);
                ++nsets;
            });
        report(state, tally);
        return nsets;
    }

    // `site` passes each set it evaluates to f(level, set), which only asks
    // whether the set is empty, with B::empty, and returns 1 if it is not, 0
    // if it is. Reports the number of sets that are not (`nonempty`).
    // Returns the number of sets of one iteration.
    template <class B, class Site>
    std::size_t run_empty_site(benchmark::State& state, Site&& site)
    {
        std::size_t nonempty = 0;
        std::size_t nsets    = 0;
        for (auto _ : state)
        {
            nonempty = 0;
            nsets    = 0;
            site(
                [&](std::size_t, const auto& set)
                {
                    bool is_empty = Algebra<B>::empty(set);
                    benchmark::DoNotOptimize(is_empty);
                    nonempty += is_empty ? 0 : 1;
                    ++nsets;
                    return is_empty ? std::size_t{0} : std::size_t{1};
                });
        }
        state.counters["nonempty"] = static_cast<double>(nonempty);
        return nsets;
    }

    //////////////////////////////////////////////////////////////////////
    // Synthetic LevelCellArrays
    //////////////////////////////////////////////////////////////////////

    enum class Shape : int
    {
        box,     // [n/4, 3n/4)^dim
        shell,   // |r - 0.3 n| < 3 cells: the footprint of a front at one level
        stripes, // the box cut into 2k slabs along one axis, every other kept
        balls,   // k balls, half the volume of the box in total
    };

    // Deterministic pseudo random numbers in [0, 1): reproducible inputs on
    // every platform, unlike the standard library distributions.
    class Lcg
    {
      public:

        explicit Lcg(std::uint64_t seed)
            : m_state(seed)
        {
        }

        double next()
        {
            m_state = m_state * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<double>(m_state >> 11) * (1.0 / 9007199254740992.0);
        }

      private:

        std::uint64_t m_state;
    };

    template <std::size_t dim>
    std::vector<std::array<double, dim>> ball_centers(std::size_t nballs, double lo, double hi)
    {
        Lcg rng(42);
        std::vector<std::array<double, dim>> centers(nballs);
        for (auto& c : centers)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                c[d] = lo + (hi - lo) * rng.next();
            }
        }
        return centers;
    }

    // The cells of [0, 2^level)^dim for which `inside` holds.
    template <std::size_t dim, class Predicate>
    samurai::LevelCellArray<dim> make_lca(std::size_t level, Predicate&& inside)
    {
        samurai::LevelCellArray<dim> lca(level);
        samurai::static_nested_loop<dim>(0,
                                         1 << level,
                                         [&](const auto& cell)
                                         {
                                             if (inside(cell))
                                             {
                                                 lca.add_point_back(cell[0], xt::view(cell, xt::range(1, dim)));
                                             }
                                         });
        return lca;
    }

    // `axis` and `k` only matter to the stripes; `k` to the balls.
    template <std::size_t dim>
    samurai::LevelCellArray<dim> make_shape(Shape shape, std::size_t level, std::size_t axis = 0, std::size_t k = 1)
    {
        using value_t   = typename samurai::LevelCellArray<dim>::value_t;
        const value_t n = value_t(1) << level;
        const double nd = static_cast<double>(n);

        const samurai::Box<value_t, dim> box(xt::ones<value_t>({dim}) * (n / 4), xt::ones<value_t>({dim}) * (3 * n / 4));

        switch (shape)
        {
            case Shape::box:
                return samurai::LevelCellArray<dim>(level, box);
            case Shape::shell:
                return make_lca<dim>(level,
                                     [nd](const auto& cell)
                                     {
                                         double r2 = 0;
                                         for (std::size_t d = 0; d < dim; ++d)
                                         {
                                             const double x = static_cast<double>(cell[d]) + 0.5 - 0.5 * nd;
                                             r2 += x * x;
                                         }
                                         return std::abs(std::sqrt(r2) - 0.3 * nd) < 3.;
                                     });
            case Shape::stripes:
            {
                const value_t slab = std::max<value_t>(1, n / (4 * static_cast<value_t>(k)));
                return make_lca<dim>(level,
                                     [=](const auto& cell)
                                     {
                                         return xt::all(cell >= box.min_corner() && cell < box.max_corner())
                                             && ((cell[axis] - n / 4) / slab) % 2 == 0;
                                     });
            }
            case Shape::balls:
            {
                const auto centers  = ball_centers<dim>(k, 0.1 * nd, 0.9 * nd);
                const double unit   = (dim == 1) ? 2. : ((dim == 2) ? std::numbers::pi : 4. * std::numbers::pi / 3.);
                const double volume = std::pow(0.5 * nd, static_cast<double>(dim)) / 2. / static_cast<double>(k);
                const double radius = std::pow(volume / unit, 1. / static_cast<double>(dim));
                return make_lca<dim>(level,
                                     [&centers, radius](const auto& cell)
                                     {
                                         for (const auto& c : centers)
                                         {
                                             double r2 = 0;
                                             for (std::size_t d = 0; d < dim; ++d)
                                             {
                                                 const double x = static_cast<double>(cell[d]) + 0.5 - c[d];
                                                 r2 += x * x;
                                             }
                                             if (r2 < radius * radius)
                                             {
                                                 return true;
                                             }
                                         }
                                         return false;
                                     });
            }
        }
        return samurai::LevelCellArray<dim>(level);
    }

    // Number of non-empty rows: the unit of the per-row work of a traversal
    // (row lookup, coarsening and expansion buffers).
    template <std::size_t dim>
    std::size_t count_rows(const samurai::LevelCellArray<dim>& lca)
    {
        if constexpr (dim == 1)
        {
            return lca.empty() ? 0 : 1;
        }
        else
        {
            std::size_t rows = 0;
            for (const auto& i : lca[1])
            {
                rows += static_cast<std::size_t>(i.size());
            }
            return rows;
        }
    }

    // The front of an adapted mesh at one level: a thin shell in 2D and 3D;
    // in 1D, where a shell has only two intervals, 4096 small segments.
    template <std::size_t dim>
    samurai::LevelCellArray<dim> make_front(std::size_t level)
    {
        return (dim == 1) ? make_shape<dim>(Shape::balls, level, 0, 4096) : make_shape<dim>(Shape::shell, level);
    }

    //////////////////////////////////////////////////////////////////////
    // The level of the synthetic sets
    //////////////////////////////////////////////////////////////////////

    // Whether the build has the sweep over the level
    // (-DSAMURAI_BENCHMARK_SET_ALGEBRA_SWEEP=ON): more levels for the
    // synthetic sets and more meshes for the call sites.
#ifdef SAMURAI_SET_ALGEBRA_SWEEP
    inline constexpr bool sweep_enabled = true;
#else
    inline constexpr bool sweep_enabled = false;
#endif

    // The level of the synthetic sets of a default run.
    template <std::size_t dim>
    constexpr std::size_t synthetic_level()
    {
        return (dim == 1) ? 20 : ((dim == 2) ? 11 : 8);
    }

    // The levels the synthetic benchmarks run at, their first argument:
    // synthetic_level, and with the sweep a range of levels around it, for
    // curves of the time against the number of intervals. In 1D the front
    // needs level 16 for its 4096 segments to be at least two cells wide.
    template <std::size_t dim>
    std::vector<std::size_t> synthetic_levels()
    {
        if constexpr (!sweep_enabled)
        {
            return {synthetic_level<dim>()};
        }
        else if constexpr (dim == 1)
        {
            return {16, 18, 20, 22, 24};
        }
        else if constexpr (dim == 2)
        {
            return {8, 9, 10, 11, 12, 13};
        }
        else
        {
            return {5, 6, 7, 8, 9};
        }
    }

    // The entry of `key` in `cache`, make() the first time.
    template <class Key, class T, class Make>
    const T& cached(std::map<Key, T>& cache, const std::type_identity_t<Key>& key, Make&& make)
    {
        auto it = cache.find(key);
        if (it == cache.end())
        {
            it = cache.emplace(key, make()).first;
        }
        return it->second;
    }

    // make_front(level), built once per binary.
    template <std::size_t dim>
    const samurai::LevelCellArray<dim>& front(std::size_t level)
    {
        static std::map<std::size_t, samurai::LevelCellArray<dim>> cache;
        return cached(cache,
                      level,
                      [&]
                      {
                          return make_front<dim>(level);
                      });
    }

    // The level of a synthetic benchmark, its first argument.
    inline std::size_t synthetic_level(const benchmark::State& state)
    {
        return static_cast<std::size_t>(state.range(0));
    }

    // Registers one benchmark per level of synthetic_levels, with the
    // arguments `more(level)` after the level: a list of argument lists,
    // none for a benchmark without other arguments.
    template <std::size_t dim, class More>
    void level_args(benchmark::internal::Benchmark* b, More&& more)
    {
        for (const std::size_t level : synthetic_levels<dim>())
        {
            for (const auto& args : more(level))
            {
                std::vector<int64_t> all{static_cast<int64_t>(level)};
                all.insert(all.end(), args.begin(), args.end());
                b->Args(all);
            }
        }
    }

    // One benchmark per level, without other arguments.
    template <std::size_t dim>
    void level_args(benchmark::internal::Benchmark* b)
    {
        level_args<dim>(b,
                        [](std::size_t)
                        {
                            return std::vector<std::vector<int64_t>>{{}};
                        });
    }

    template <std::size_t dim>
    void report_input(benchmark::State& state, const samurai::LevelCellArray<dim>& lca)
    {
        state.counters["in_rows"] = static_cast<double>(count_rows(lca));
        state.counters["in_itv"]  = static_cast<double>(lca[0].size());
    }

    enum class Reach
    {
        unit,    // +-e_d: the operands overlap almost entirely
        growing, // +-r e_d, r growing every 2 dim operands: all operands differ
    };

    // The directions of the star stencil, +-e_0, +-e_1, ..., in turn.
    template <std::size_t dim, std::size_t N>
    std::array<samurai::DirectionVector<dim>, N> translations(Reach reach)
    {
        std::array<samurai::DirectionVector<dim>, N> t;
        std::size_t i = 0;
        for (int radius = 1; i < N; radius += (reach == Reach::growing) ? 1 : 0)
        {
            samurai::for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    if (i < N)
                    {
                        t[i++] = radius * direction;
                    }
                });
        }
        return t;
    }

    // The stencil pattern of the library.
    template <class Op, class LCA, class Translations, std::size_t... Is>
    auto make_nary(Op&& op, const LCA& lca, const Translations& t, std::index_sequence<Is...>)
    {
        return op(samurai::self(lca), samurai::translate(lca, t[Is])...);
    }

    inline constexpr auto intersection_op = [](const auto&... s)
    {
        return samurai::intersection(s...);
    };
    inline constexpr auto union_op = [](const auto&... s)
    {
        return samurai::union_(s...);
    };
} // namespace bench
