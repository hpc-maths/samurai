// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Inputs and helpers shared by the set algebra benchmarks: synthetic
// LevelCellArrays at one level, each shape chosen to stress one property of
// the traversal, and MR-adapted meshes refined around the surfaces of balls.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numbers>
#include <tuple>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <xtensor/core/xoperation.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/views/xview.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/stencil.hpp>
#include <samurai/subset/node.hpp>

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
    // the traversal of a lazy expression.
    template <class Set>
    Tally traverse(const Set& set)
    {
        Tally tally;
        set(
            [&tally](const auto& i, const auto& index)
            {
                tally(i, index);
            });
        benchmark::DoNotOptimize(tally);
        return tally;
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

    // `site` passes each set it evaluates to f(level, set). Returns the number
    // of sets of one iteration.
    template <class Site>
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
                    tally += traverse(set);
                    ++nsets;
                });
        }
        report(state, tally);
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

    //////////////////////////////////////////////////////////////////////
    // MR-adapted meshes
    //////////////////////////////////////////////////////////////////////

    template <std::size_t dim>
    constexpr std::size_t default_max_level()
    {
        return (dim == 1) ? 16 : ((dim == 2) ? 10 : 7);
    }

    template <std::size_t dim>
    auto mr_config(std::size_t max_level)
    {
        return samurai::mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(2).disable_minimal_ghost_width();
    }

    template <std::size_t dim>
    using mr_mesh_t = decltype(samurai::mra::make_mesh(std::declval<samurai::Box<double, dim>>(), mr_config<dim>(4)));

    template <std::size_t dim>
    using mr_field_t = decltype(samurai::make_scalar_field<double>("u", std::declval<mr_mesh_t<dim>&>()));

    // The mesh adapted to this indicator is refined to the finest level along
    // every ball surface. `shift` moves the balls along the diagonal.
    template <std::size_t dim>
    auto balls_indicator(std::size_t nballs, double shift)
    {
        return [centers = ball_centers<dim>(nballs, 0.25, 0.75),
                radius  = 0.2 / std::pow(static_cast<double>(nballs), 1. / static_cast<double>(dim)),
                shift](const auto& x)
        {
            for (const auto& c : centers)
            {
                double r2 = 0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const double dx = x[d] - c[d] - shift;
                    r2 += dx * dx;
                }
                if (r2 <= radius * radius)
                {
                    return 1.;
                }
            }
            return 0.;
        };
    }

    constexpr double mr_epsilon = 1e-3;

    // How far the fronts move in one time step.
    template <class Mesh>
    double step_shift(const Mesh& mesh)
    {
        return 4. * mesh.min_cell_length();
    }

    template <std::size_t dim>
    mr_mesh_t<dim> make_unit_mesh(std::size_t max_level)
    {
        return samurai::mra::make_mesh(samurai::Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})),
                                       mr_config<dim>(max_level));
    }

    template <std::size_t dim>
    struct AdaptedCase
    {
        mr_mesh_t<dim> mesh;
        mr_field_t<dim> u;

        AdaptedCase(std::size_t max_level, std::size_t nballs, bool moved)
            : mesh(make_unit_mesh<dim>(max_level))
            , u(samurai::make_scalar_field<double>("u", mesh, balls_indicator<dim>(nballs, moved ? step_shift(mesh) : 0.)))
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
            samurai::make_MRAdapt(u)(samurai::mra_config().epsilon(mr_epsilon));
        }

        AdaptedCase(const AdaptedCase&)            = delete;
        AdaptedCase& operator=(const AdaptedCase&) = delete;
    };

    // Built once per binary and shared by its benchmarks. `moved` gives the
    // mesh of the next time step.
    template <std::size_t dim>
    AdaptedCase<dim>& adapted_case(std::size_t max_level, std::size_t nballs, bool moved = false)
    {
        static std::map<std::tuple<std::size_t, std::size_t, bool>, std::unique_ptr<AdaptedCase<dim>>> cache;
        auto& entry = cache[{max_level, nballs, moved}];
        if (!entry)
        {
            entry = std::make_unique<AdaptedCase<dim>>(max_level, nballs, moved);
        }
        return *entry;
    }

    // So that every report says what it was measured on.
    template <class Mesh>
    void report_mesh(benchmark::State& state, const Mesh& mesh)
    {
        using mesh_id_t       = typename Mesh::mesh_id_t;
        std::size_t intervals = 0;
        samurai::for_each_interval(mesh[mesh_id_t::cells],
                                   [&](std::size_t, const auto&, const auto&)
                                   {
                                       ++intervals;
                                   });
        state.counters["leaves"]   = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
        state.counters["leaf_itv"] = static_cast<double>(intervals);
        state.counters["levels"]   = static_cast<double>(mesh[mesh_id_t::cells].max_level() - mesh[mesh_id_t::cells].min_level() + 1);
    }

} // namespace bench
