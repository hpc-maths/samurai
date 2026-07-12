// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Cost of the set-algebra primitives (intersection, union, translate,
// on()) that every mesh traversal in the library is built from
// (include/samurai/subset/). Two tiers:
//
//   - "uniform": the raw operator cost on three fixed, overlapping boxes at a
//     single level (Box::box construction, no MR mesh). Each subset is
//     iterated and its interval sizes accumulated into a DoNotOptimize'd
//     variable, so the lazy expression is actually evaluated -- an
//     unconsumed expression is legal C++ and the compiler is free to remove
//     it entirely.
//   - "adapted": the same operators applied across levels of a genuinely
//     MR-adapted mesh (intersection(all_cells[l], cells[l+1]).on(l), the
//     projection footprint used by compute_detail and update_ghost_mr; and
//     intersection(cells[l], translate(cells[l], stencil)), the pattern
//     behind ghost filling). A static uniform mesh does not exercise
//     cross-level traversal and is not representative of mesh adaptation,
//     which is ~62% of the advection_2d baseline (Improvement/perf-baseline.md).
//
// A prior attempt to replace this set-algebra engine with a "batch"
// evaluation strategy won on tier-1-style micro-benchmarks but was 24-32%
// slower on the end-to-end advection_2d run; the "adapted" tier exists so
// that this class of regression is visible without having to rebuild and run
// a full demo.

#include <cmath>
#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/subset/nary_set_operator.hpp>
#include <samurai/subset/node.hpp>

namespace
{
    template <std::size_t dim>
    auto make_three_boxes()
    {
        constexpr std::size_t level = 8;
        samurai::Box<int, dim> box1({0, 0}, {1 << level, 1 << level});
        samurai::Box<int, dim> box2({0, 0}, {(1 << (level - 1)), (1 << (level - 1))});
        samurai::Box<int, dim> box3({1, 1}, {(1 << (level - 1)) + 1, (1 << (level - 1)) + 1});
        return std::make_tuple(samurai::LevelCellArray<dim>{level, box1},
                               samurai::LevelCellArray<dim>{level, box2},
                               samurai::LevelCellArray<dim>{level, box3});
    }

    // Deterministic adapted mesh: MR adaptation of a tanh front. Reproducible
    // (no RNG) and representative of a real multi-level mesh.
    template <std::size_t dim>
    auto make_adapted_mesh(double eps, std::size_t max_level)
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(0);
        max_corner.fill(1);
        const samurai::Box<double, dim> box(min_corner, max_corner);

        auto config = samurai::mesh_config<dim>().min_level(4).max_level(max_level).max_stencil_size(2).disable_minimal_ghost_width();
        auto mesh   = samurai::mra::make_mesh(box, config);

        // Interior blob: the refined region must not reach the domain boundary
        // (a boundary-touching refinement crashes the ghost sweep on a 3D mesh).
        auto u = samurai::make_scalar_field<double>("u", mesh);
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   auto c   = cell.center();
                                   double r = 0;
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       r += (c[d] - 0.3) * (c[d] - 0.3);
                                   }
                                   u[cell] = (r <= 0.04) ? 1. : 0.;
                               });
        samurai::make_MRAdapt(u)(samurai::mra_config().epsilon(eps));
        return mesh;
    }
} // namespace

// --- Tier 1: raw operator on fixed uniform boxes ---------------------------

static void BM_Intersection3_uniform(benchmark::State& state)
{
    constexpr std::size_t dim = 2;
    auto [set1, set2, set3]   = make_three_boxes<dim>();
    for (auto _ : state)
    {
        std::size_t acc = 0;
        auto subset     = samurai::intersection(samurai::intersection(set1, set2), set3);
        subset(
            [&](const auto& i, const auto&)
            {
                acc += i.size();
            });
        benchmark::DoNotOptimize(acc);
    }
}

static void BM_Intersection3_uniform_on(benchmark::State& state)
{
    constexpr std::size_t dim = 2;
    auto [set1, set2, set3]   = make_three_boxes<dim>();
    for (auto _ : state)
    {
        std::size_t acc = 0;
        auto subset     = samurai::intersection(samurai::intersection(set1, set2), set3).on(7);
        subset(
            [&](const auto& i, const auto&)
            {
                acc += i.size();
            });
        benchmark::DoNotOptimize(acc);
    }
}

static void BM_TranslatedIntersection_uniform(benchmark::State& state)
{
    constexpr std::size_t dim                       = 2;
    auto [set1, set2, set3]                         = make_three_boxes<dim>();
    xt::xtensor_fixed<int, xt::xshape<dim>> stencil = {1, 0};
    for (auto _ : state)
    {
        std::size_t acc = 0;
        auto subset     = samurai::intersection(set1, samurai::translate(set2, stencil));
        subset(
            [&](const auto& i, const auto&)
            {
                acc += i.size();
            });
        benchmark::DoNotOptimize(acc);
    }
}

BENCHMARK(BM_Intersection3_uniform);
BENCHMARK(BM_Intersection3_uniform_on);
BENCHMARK(BM_TranslatedIntersection_uniform);

// --- Tier 2: cross-level set algebra on an adapted mesh --------------------

// intersection(all_cells[l], cells[l+1]).on(l): the projection footprint used
// by detail computation and ghost update. Arg: 1/eps.
template <std::size_t dim>
static void BM_ProjectionFootprint_adapted(benchmark::State& state)
{
    const std::size_t max_level = (dim == 2) ? 11 : 8;
    auto mesh                   = make_adapted_mesh<dim>(1. / static_cast<double>(state.range(0)), max_level);
    using mesh_id_t             = typename std::decay_t<decltype(mesh)>::mesh_id_t;

    const auto min_l = mesh[mesh_id_t::cells].min_level();
    const auto max_l = mesh[mesh_id_t::cells].max_level();

    for (auto _ : state)
    {
        std::size_t acc = 0;
        for (std::size_t level = min_l; level < max_l; ++level)
        {
            auto subset = samurai::intersection(mesh[mesh_id_t::all_cells][level], mesh[mesh_id_t::cells][level + 1]).on(level);
            subset(
                [&](const auto& i, const auto&)
                {
                    acc += i.size();
                });
        }
        benchmark::DoNotOptimize(acc);
    }
    state.counters["cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
}

// translate + intersect per level: the stencil pattern of ghost filling.
template <std::size_t dim>
static void BM_StencilTranslation_adapted(benchmark::State& state)
{
    const std::size_t max_level = (dim == 2) ? 11 : 8;
    auto mesh                   = make_adapted_mesh<dim>(1. / static_cast<double>(state.range(0)), max_level);
    using mesh_id_t             = typename std::decay_t<decltype(mesh)>::mesh_id_t;

    const auto min_l = mesh[mesh_id_t::cells].min_level();
    const auto max_l = mesh[mesh_id_t::cells].max_level();

    xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
    stencil.fill(0);
    stencil[0] = 1;

    for (auto _ : state)
    {
        std::size_t acc = 0;
        for (std::size_t level = min_l; level <= max_l; ++level)
        {
            auto subset = samurai::intersection(mesh[mesh_id_t::cells][level], samurai::translate(mesh[mesh_id_t::cells][level], stencil));
            subset(
                [&](const auto& i, const auto&)
                {
                    acc += i.size();
                });
        }
        benchmark::DoNotOptimize(acc);
    }
    state.counters["cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
}

BENCHMARK(BM_ProjectionFootprint_adapted<2>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_ProjectionFootprint_adapted<3>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_StencilTranslation_adapted<2>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_StencilTranslation_adapted<3>)->Arg(1000)->Arg(100000);

BENCHMARK_MAIN();
