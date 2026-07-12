#include <array>
#include <cmath>

#include <benchmark/benchmark.h>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil_field.hpp>

// Finite-volume scheme evaluation. In the advection_2d baseline the field
// expression `unp1 = u - dt*upwind(a,u)` (the "field expressions" timer) is the
// single largest timed poste after mesh adaptation (~29% of the long run), yet no
// micro-benchmark exercised it. The ghost update is done once before the timed
// loop so only the flux evaluation and field arithmetic are measured.

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

BENCHMARK_MAIN();
