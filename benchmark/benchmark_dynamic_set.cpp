// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Static vs dynamic set algebra: the price of the runtime version.
//
// The static algebra (include/samurai/subset/) encodes the whole expression
// in the C++ type system: the traversal is fully inlined, no virtual calls,
// no heap. The dynamic algebra (include/samurai/subset/dynamic/) trades that
// for a tree decided at runtime: each node is heap-allocated (shared_ptr) and
// every interval flows through a virtual ISetTraverser call.
//
// Each pattern below is measured in both forms, side by side, so the ratio is
// read directly off the report (compare BM_*_static vs BM_*_dynamic). Both
// build the expression inside the timed loop, matching benchmark_set.cpp and
// the realistic usage where the mesh - hence the set - changes every step. So
// the dynamic figure is the *end-to-end* cost: node construction + virtual
// traversal, which is what a caller actually pays.
//
// Tiers mirror benchmark_set.cpp:
//   - "uniform": raw operator cost on three fixed overlapping boxes;
//   - "adapted": the cross-level projection / stencil footprints on a real
//     MR-adapted mesh, the patterns that dominate mesh adaptation.

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
#include <samurai/subset/dynamic/dynamic.hpp>
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

    // Iterate a set and accumulate interval sizes into a sink so the lazy
    // expression is actually evaluated.
    template <class Set>
    std::size_t consume(const Set& set)
    {
        std::size_t acc = 0;
        set(
            [&](const auto& i, const auto&)
            {
                acc += i.size();
            });
        return acc;
    }
} // namespace

// --- Tier 1: raw operator on fixed uniform boxes ---------------------------

static void BM_Intersection3_uniform_static(benchmark::State& state)
{
    auto [s1, s2, s3] = make_three_boxes<2>();
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(consume(samurai::intersection(samurai::intersection(s1, s2), s3)));
    }
}

static void BM_Intersection3_uniform_dynamic(benchmark::State& state)
{
    auto [s1, s2, s3] = make_three_boxes<2>();
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(consume(samurai::dyn::intersection(samurai::dyn::self(s1), samurai::dyn::self(s2), samurai::dyn::self(s3))));
    }
}

static void BM_Intersection3_on_uniform_static(benchmark::State& state)
{
    auto [s1, s2, s3] = make_three_boxes<2>();
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(consume(samurai::intersection(samurai::intersection(s1, s2), s3).on(7)));
    }
}

static void BM_Intersection3_on_uniform_dynamic(benchmark::State& state)
{
    auto [s1, s2, s3] = make_three_boxes<2>();
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            consume(samurai::dyn::intersection(samurai::dyn::self(s1), samurai::dyn::self(s2), samurai::dyn::self(s3)).on(7)));
    }
}

static void BM_TranslatedIntersection_uniform_static(benchmark::State& state)
{
    auto [s1, s2, s3]                          = make_three_boxes<2>();
    xt::xtensor_fixed<int, xt::xshape<2>> sten = {1, 0};
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(consume(samurai::intersection(s1, samurai::translate(s2, sten))));
    }
}

static void BM_TranslatedIntersection_uniform_dynamic(benchmark::State& state)
{
    auto [s1, s2, s3]                          = make_three_boxes<2>();
    xt::xtensor_fixed<int, xt::xshape<2>> sten = {1, 0};
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            consume(samurai::dyn::intersection(samurai::dyn::self(s1), samurai::dyn::translate(samurai::dyn::self(s2), sten))));
    }
}

BENCHMARK(BM_Intersection3_uniform_static);
BENCHMARK(BM_Intersection3_uniform_dynamic);
BENCHMARK(BM_Intersection3_on_uniform_static);
BENCHMARK(BM_Intersection3_on_uniform_dynamic);
BENCHMARK(BM_TranslatedIntersection_uniform_static);
BENCHMARK(BM_TranslatedIntersection_uniform_dynamic);

// --- Tier 2: cross-level set algebra on an adapted mesh --------------------

// intersection(all_cells[l], cells[l+1]).on(l): projection footprint.
template <std::size_t dim>
static void BM_ProjectionFootprint_adapted_static(benchmark::State& state)
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
            acc += consume(samurai::intersection(mesh[mesh_id_t::all_cells][level], mesh[mesh_id_t::cells][level + 1]).on(level));
        }
        benchmark::DoNotOptimize(acc);
    }
    state.counters["cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
}

template <std::size_t dim>
static void BM_ProjectionFootprint_adapted_dynamic(benchmark::State& state)
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
            acc += consume(samurai::dyn::intersection(samurai::dyn::self(mesh[mesh_id_t::all_cells][level]),
                                                      samurai::dyn::self(mesh[mesh_id_t::cells][level + 1]))
                               .on(level));
        }
        benchmark::DoNotOptimize(acc);
    }
    state.counters["cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
}

// intersection(cells[l], translate(cells[l], stencil)): ghost-filling stencil.
template <std::size_t dim>
static void BM_StencilTranslation_adapted_static(benchmark::State& state)
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
            acc += consume(samurai::intersection(mesh[mesh_id_t::cells][level], samurai::translate(mesh[mesh_id_t::cells][level], stencil)));
        }
        benchmark::DoNotOptimize(acc);
    }
    state.counters["cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
}

template <std::size_t dim>
static void BM_StencilTranslation_adapted_dynamic(benchmark::State& state)
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
            acc += consume(samurai::dyn::intersection(samurai::dyn::self(mesh[mesh_id_t::cells][level]),
                                                      samurai::dyn::translate(samurai::dyn::self(mesh[mesh_id_t::cells][level]), stencil)));
        }
        benchmark::DoNotOptimize(acc);
    }
    state.counters["cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
}

BENCHMARK(BM_ProjectionFootprint_adapted_static<2>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_ProjectionFootprint_adapted_dynamic<2>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_ProjectionFootprint_adapted_static<3>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_ProjectionFootprint_adapted_dynamic<3>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_StencilTranslation_adapted_static<2>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_StencilTranslation_adapted_dynamic<2>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_StencilTranslation_adapted_static<3>)->Arg(1000)->Arg(100000);
BENCHMARK(BM_StencilTranslation_adapted_dynamic<3>)->Arg(1000)->Arg(100000);

BENCHMARK_MAIN();
