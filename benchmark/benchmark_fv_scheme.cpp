// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Evaluation cost of the linear-advection operator `u - dt * conv(u)`, on
// an MR-adapted mesh (an interior spherical blob refined by epsilon; see
// make_adapted_mesh), compared across two independent implementations of
// the same operator: identical mesh, velocity vector (1 in every Cartesian
// direction) and dt.
//
//   - benchmark_upwind: samurai::upwind(a, u), the direct field-expression
//     stencil (stencil_field.hpp) used by demos/FiniteVolume/advection_2d.cpp.
//   - benchmark_convection_upwind_flux: samurai::make_convection_upwind
//     <Field>(velocity), the flux-based scheme framework (FluxDefinition /
//     FluxConfig in schemes/fv/operators/convection_lin.hpp), the same
//     construction used by demos/FiniteVolume/linear_convection.cpp.
//
// In the advection_2d baseline, the field expression built on
// samurai::upwind() (the "field expressions" timer) is the single largest
// timed item after mesh adaptation, ~29% of the long run
// (Improvement/perf-baseline.md), yet was not covered by any
// micro-benchmark before this file. The ghost update is done once before
// the timed loop in both benchmarks, so only flux evaluation and field
// arithmetic are measured, not ghost filling.

#include <array>
#include <cmath>

#include <benchmark/benchmark.h>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/stencil_field.hpp>

namespace
{
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

        // Interior blob: the refined region must not reach the domain boundary.
        // A refinement touching the boundary/corner makes update_ghost_mr read a
        // non-existent ghost interval on a 3D mesh (see NOTE below).
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

// Arg: 1/eps.
template <std::size_t dim>
void benchmark_upwind(benchmark::State& state)
{
    const std::size_t max_level = (dim == 2) ? 11 : 8;
    auto mesh                   = make_adapted_mesh<dim>(1. / static_cast<double>(state.range(0)), max_level);
    using mesh_id_t             = typename std::decay_t<decltype(mesh)>::mesh_id_t;

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               u[cell] = static_cast<double>(cell.center(0));
                           });

    std::array<double, dim> a;
    a.fill(1.0);
    const double dt = 0.1;

    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    samurai::update_ghost_mr(u);

    for (auto _ : state)
    {
        unp1 = u - dt * samurai::upwind(a, u);
        benchmark::DoNotOptimize(unp1.array().data());
        benchmark::ClobberMemory();
    }

    state.counters["cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
    state.SetItemsProcessed(static_cast<int64_t>(mesh.nb_cells(mesh_id_t::cells)) * state.iterations());
}

BENCHMARK(benchmark_upwind<2>)->Unit(benchmark::kMillisecond)->Arg(1000)->Arg(100000);
BENCHMARK(benchmark_upwind<3>)->Unit(benchmark::kMillisecond)->Arg(1000)->Arg(100000);

// Same problem as benchmark_upwind, evaluated through the flux-based scheme
// framework instead of the direct field expression. Arg: 1/eps.
template <std::size_t dim>
void benchmark_convection_upwind_flux(benchmark::State& state)
{
    const std::size_t max_level = (dim == 2) ? 11 : 8;
    auto mesh                   = make_adapted_mesh<dim>(1. / static_cast<double>(state.range(0)), max_level);
    using mesh_id_t             = typename std::decay_t<decltype(mesh)>::mesh_id_t;

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               u[cell] = static_cast<double>(cell.center(0));
                           });

    samurai::VelocityVector<dim> velocity;
    velocity.fill(1.0);
    const double dt = 0.1;

    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    samurai::update_ghost_mr(u);

    auto conv = samurai::make_convection_upwind<decltype(u)>(velocity);

    for (auto _ : state)
    {
        unp1 = u - dt * conv(u);
        benchmark::DoNotOptimize(unp1.array().data());
        benchmark::ClobberMemory();
    }

    state.counters["cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
    state.SetItemsProcessed(static_cast<int64_t>(mesh.nb_cells(mesh_id_t::cells)) * state.iterations());
}

BENCHMARK(benchmark_convection_upwind_flux<2>)->Unit(benchmark::kMillisecond)->Arg(1000)->Arg(100000);
BENCHMARK(benchmark_convection_upwind_flux<3>)->Unit(benchmark::kMillisecond)->Arg(1000)->Arg(100000);

BENCHMARK_MAIN();
