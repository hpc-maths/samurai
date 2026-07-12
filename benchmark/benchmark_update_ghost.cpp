#include <cmath>

#include <benchmark/benchmark.h>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

// update_ghost_mr is the projection/prediction sweep that fills the ghost cells
// of an adapted MR mesh. In the advection_2d baseline it accounts for ~26% of the
// total runtime (adaptation + time loop), so it is measured here on a genuinely
// adapted mesh rather than a static uniform one.

namespace
{
    template <class Field>
    void fill_field(Field& u, std::size_t nb, std::size_t direction)
    {
        samurai::for_each_cell(u.mesh(),
                               [&](auto& cell)
                               {
                                   auto center = cell.center();
                                   u[cell]     = 0;
                                   for (std::size_t i = 1; i <= nb; ++i)
                                   {
                                       u[cell] += std::tanh(
                                           1000 * std::abs(center[direction] - static_cast<double>(i) / static_cast<double>(nb + 1)));
                                   }
                                   u[cell] -= static_cast<double>(nb);
                               });
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

        auto adaptation = samurai::make_MRAdapt(u);
        adaptation(samurai::mra_config().epsilon(eps));

        return mesh;
    }
} // namespace

// Args: 1/eps, number of fields updated together (1 or 4).
template <std::size_t dim>
void benchmark_update_ghost(benchmark::State& state)
{
    const double eps            = 1. / static_cast<double>(state.range(0));
    const std::size_t nfields   = static_cast<std::size_t>(state.range(1));
    const std::size_t max_level = (dim == 2) ? 11 : 8;

    auto mesh = make_adapted_mesh<dim>(eps, max_level);

    auto u1 = samurai::make_scalar_field<double>("u1", mesh);
    auto u2 = samurai::make_scalar_field<double>("u2", mesh);
    auto u3 = samurai::make_scalar_field<double>("u3", mesh);
    auto u4 = samurai::make_scalar_field<double>("u4", mesh);
    fill_field(u1, 2, 0);
    fill_field(u2, 3, 0);
    fill_field(u3, 2, (dim > 1) ? 1 : 0);
    fill_field(u4, 3, (dim > 1) ? 1 : 0);

    samurai::make_bc<samurai::Dirichlet<1>>(u1, 0.);
    samurai::make_bc<samurai::Dirichlet<1>>(u2, 0.);
    samurai::make_bc<samurai::Dirichlet<1>>(u3, 0.);
    samurai::make_bc<samurai::Dirichlet<1>>(u4, 0.);

    using mesh_id_t = typename std::decay_t<decltype(mesh)>::mesh_id_t;

    for (auto _ : state)
    {
        if (nfields == 1)
        {
            samurai::update_ghost_mr(u1);
            benchmark::DoNotOptimize(u1.array().data());
        }
        else
        {
            samurai::update_ghost_mr(u1, u2, u3, u4);
            benchmark::DoNotOptimize(u1.array().data());
            benchmark::DoNotOptimize(u4.array().data());
        }
        benchmark::ClobberMemory();
    }

    state.counters["cells"]     = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
    state.counters["all_cells"] = static_cast<double>(mesh.nb_cells(mesh_id_t::reference));
}

std::vector<std::vector<int64_t>> ghost_args = {
    {1000, 100000},
    {1,    4     }
};

BENCHMARK(benchmark_update_ghost<2>)->Unit(benchmark::kMillisecond)->ArgsProduct(ghost_args);
BENCHMARK(benchmark_update_ghost<3>)->Unit(benchmark::kMillisecond)->ArgsProduct(ghost_args);

BENCHMARK_MAIN();
