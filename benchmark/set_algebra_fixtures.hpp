// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Fixtures shared by the set algebra benchmarks (benchmark_set_*.cpp).
//
// Two families of inputs:
//
//   - synthetic LevelCellArrays at one level, whose shape is chosen to stress
//     one property of the traversal at a time: a compact box, a thin spherical
//     shell (the footprint of a front at one level of an adapted mesh), a box
//     cut into K stripes along a chosen axis (fragmentation along that axis at
//     a constant number of cells) and K separated balls;
//   - MR-adapted meshes in 1D, 2D and 3D, refined around the surfaces of K
//     balls, built once per binary and shared by its benchmarks.
//
// Every traversal goes through a Tally, which consumes the intervals emitted
// by the set: an unconsumed lazy expression is legal C++ and the compiler is
// free to remove it.

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

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
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

    // Traverses the set once and returns what it emitted.
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

    // Output size of one iteration, and the time per emitted interval: the
    // figure to compare across shapes, operators and dimensions.
    inline void report(benchmark::State& state, const Tally& tally)
    {
        state.counters["itv"]   = static_cast<double>(tally.intervals);
        state.counters["cells"] = static_cast<double>(tally.cells);
        state.counters["t/itv"] = benchmark::Counter(static_cast<double>(tally.intervals),
                                                     benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    }

    //////////////////////////////////////////////////////////////////////
    // Synthetic LevelCellArrays
    //////////////////////////////////////////////////////////////////////

    enum class Shape : int
    {
        box,     // [n/4, 3n/4)^dim
        shell,   // |r - 0.3 n| < 3 cells: one front at one level
        stripes, // the box, keeping k stripes along one axis (half its cells)
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

    // The LevelCellArray of the cells of [0, 2^level)^dim for which `inside`
    // holds, built row by row in the order add_interval_back expects
    // (outermost coordinate slowest).
    template <std::size_t dim, class Predicate>
    samurai::LevelCellArray<dim> make_lca(std::size_t level, Predicate&& inside)
    {
        using lca_t   = samurai::LevelCellArray<dim>;
        using value_t = typename lca_t::value_t;

        const value_t n = value_t(1) << level;
        lca_t lca(level);

        std::array<value_t, dim> cell{};
        xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> yz;

        auto scan_row = [&]()
        {
            for (std::size_t d = 1; d < dim; ++d)
            {
                yz[d - 1] = cell[d];
            }
            value_t start = -1;
            for (cell[0] = 0; cell[0] <= n; ++cell[0])
            {
                const bool in = (cell[0] < n) && inside(cell);
                if (in && start < 0)
                {
                    start = cell[0];
                }
                else if (!in && start >= 0)
                {
                    lca.add_interval_back({start, cell[0]}, yz);
                    start = -1;
                }
            }
        };

        if constexpr (dim == 1)
        {
            scan_row();
        }
        else if constexpr (dim == 2)
        {
            for (cell[1] = 0; cell[1] < n; ++cell[1])
            {
                scan_row();
            }
        }
        else
        {
            for (cell[2] = 0; cell[2] < n; ++cell[2])
            {
                for (cell[1] = 0; cell[1] < n; ++cell[1])
                {
                    scan_row();
                }
            }
        }
        return lca;
    }

    // The box [lo, hi) at `level`, built from its rows only.
    template <std::size_t dim>
    samurai::LevelCellArray<dim> make_box(std::size_t level,
                                          const std::array<typename samurai::LevelCellArray<dim>::value_t, dim>& lo,
                                          const std::array<typename samurai::LevelCellArray<dim>::value_t, dim>& hi)
    {
        using value_t = typename samurai::LevelCellArray<dim>::value_t;
        samurai::LevelCellArray<dim> lca(level);
        xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> yz;
        if constexpr (dim == 1)
        {
            lca.add_interval_back({lo[0], hi[0]}, yz);
        }
        else if constexpr (dim == 2)
        {
            for (yz[0] = lo[1]; yz[0] < hi[1]; ++yz[0])
            {
                lca.add_interval_back({lo[0], hi[0]}, yz);
            }
        }
        else
        {
            for (yz[1] = lo[2]; yz[1] < hi[2]; ++yz[1])
            {
                for (yz[0] = lo[1]; yz[0] < hi[1]; ++yz[0])
                {
                    lca.add_interval_back({lo[0], hi[0]}, yz);
                }
            }
        }
        return lca;
    }

    // `axis` and `k` are read by the stripes (axis, number of stripes) and
    // `k` by the balls (number of balls).
    template <std::size_t dim>
    samurai::LevelCellArray<dim> make_shape(Shape shape, std::size_t level, std::size_t axis = 0, std::size_t k = 1)
    {
        using value_t   = typename samurai::LevelCellArray<dim>::value_t;
        const value_t n = value_t(1) << level;
        const double nd = static_cast<double>(n);

        auto in_box = [n](const auto& cell)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (cell[d] < n / 4 || cell[d] >= 3 * n / 4)
                {
                    return false;
                }
            }
            return true;
        };

        switch (shape)
        {
            case Shape::box:
                return make_lca<dim>(level, in_box);
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
                // n/2 cells along `axis`, cut into 2k slabs, the even ones kept
                const value_t slab = std::max<value_t>(1, n / (4 * static_cast<value_t>(k)));
                return make_lca<dim>(level,
                                     [=](const auto& cell)
                                     {
                                         return in_box(cell) && ((cell[axis] - n / 4) / slab) % 2 == 0;
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

    // Number of non-empty rows (distinct yz) of a LevelCellArray: the unit of
    // the per-row work of a traversal (row lookup, projection and expansion
    // buffers). Each interval of lca[1] spans that many rows.
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

    template <std::size_t dim>
    void report_input(benchmark::State& state, const samurai::LevelCellArray<dim>& lca)
    {
        state.counters["in_rows"] = static_cast<double>(count_rows(lca));
        state.counters["in_itv"]  = static_cast<double>(lca[0].size());
    }

    // `value` times the unit vector along `axis`.
    template <std::size_t dim>
    xt::xtensor_fixed<int, xt::xshape<dim>> unit(std::size_t axis, int value = 1)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> v;
        v.fill(0);
        v[axis] = value;
        return v;
    }

    //////////////////////////////////////////////////////////////////////
    // MR-adapted meshes
    //////////////////////////////////////////////////////////////////////

    // Default finest level per dimension.
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

    // Indicator of the union of `nballs` balls: the mesh adapted to it is
    // refined to the finest level along every ball surface. `shift` moves the
    // balls along the diagonal.
    template <std::size_t dim, class Field>
    void init_balls(Field& u, std::size_t nballs, double shift = 0.)
    {
        const auto centers  = ball_centers<dim>(nballs, 0.25, 0.75);
        const double radius = 0.2 / std::pow(static_cast<double>(nballs), 1. / static_cast<double>(dim));

        u.resize();
        samurai::for_each_cell(u.mesh(),
                               [&](const auto& cell)
                               {
                                   const auto x = cell.center();
                                   bool in      = false;
                                   for (const auto& c : centers)
                                   {
                                       double r2 = 0;
                                       for (std::size_t d = 0; d < dim; ++d)
                                       {
                                           const double dx = x[d] - c[d] - shift;
                                           r2 += dx * dx;
                                       }
                                       in = in || (r2 <= radius * radius);
                                   }
                                   u[cell] = in ? 1. : 0.;
                               });
    }

    constexpr double mr_epsilon = 1e-3;

    // One time step of the fronts: 4 finest cells along the diagonal.
    template <std::size_t dim>
    double step_shift(std::size_t max_level)
    {
        return 4. / static_cast<double>(std::size_t(1) << max_level);
    }

    template <std::size_t dim>
    mr_mesh_t<dim> make_unit_mesh(std::size_t max_level)
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(0);
        max_corner.fill(1);
        return samurai::mra::make_mesh(samurai::Box<double, dim>(min_corner, max_corner), mr_config<dim>(max_level));
    }

    template <std::size_t dim>
    struct AdaptedCase
    {
        mr_mesh_t<dim> mesh;
        mr_field_t<dim> u;

        AdaptedCase(std::size_t max_level, std::size_t nballs, double shift)
            : mesh(make_unit_mesh<dim>(max_level))
            , u(samurai::make_scalar_field<double>("u", mesh))
        {
            init_balls<dim>(u, nballs, shift);
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
            samurai::make_MRAdapt(u)(samurai::mra_config().epsilon(mr_epsilon));
        }

        AdaptedCase(const AdaptedCase&)            = delete;
        AdaptedCase& operator=(const AdaptedCase&) = delete;
    };

    // The adapted mesh of (max_level, nballs), built once per binary. `moved`
    // gives the mesh of the next time step (balls shifted by step_shift).
    template <std::size_t dim>
    AdaptedCase<dim>& adapted_case(std::size_t max_level, std::size_t nballs, bool moved = false)
    {
        static std::map<std::tuple<std::size_t, std::size_t, bool>, std::unique_ptr<AdaptedCase<dim>>> cache;
        auto& entry = cache[{max_level, nballs, moved}];
        if (!entry)
        {
            entry = std::make_unique<AdaptedCase<dim>>(max_level, nballs, moved ? step_shift<dim>(max_level) : 0.);
        }
        return *entry;
    }

    // Leaves, leaf intervals and levels of an adapted mesh, so that every
    // report says what it was measured on.
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
