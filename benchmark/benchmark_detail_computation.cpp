#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

#include <benchmark/benchmark.h>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/operators.hpp>
#include <samurai/subset/nary_set_operator.hpp>

template <std::size_t dim>
auto init_mesh(double eps, std::size_t direction, std::size_t nb)
{
    std::size_t min_level   = 4;
    std::size_t start_level = 8;
    std::size_t max_level   = (dim == 2) ? 12 : 8;
    std::size_t jump        = max_level - start_level;

    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
    min_corner.fill(0);
    max_corner.fill(1);
    const samurai::Box<double, dim> box(min_corner, max_corner);
    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(start_level).max_stencil_size(2).disable_minimal_ghost_width();
    auto mesh   = samurai::mra::make_mesh(box, config);
    auto u      = samurai::make_scalar_field<double>("u", mesh);

    auto init_fct = [&](auto& cell)
    {
        auto center = cell.center();
        u[cell]     = 0;
        for (std::size_t i = 1; i <= nb; ++i)
        {
            u[cell] += tanh(1000 * std::abs(center[direction] - static_cast<double>(i) / static_cast<double>(nb + 1)));
        }
        u[cell] -= static_cast<double>(nb);
    };

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               init_fct(cell);
                           });

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(eps);
    MRadaptation(mra_config);

    using cl_type = std::decay_t<decltype(mesh)>::cl_type;
    while (jump > 0)
    {
        std::cout << "MR mesh adaptation " << jump << std::endl;
        cl_type cl;
        for_each_interval(mesh,
                          [&](std::size_t level, const auto& i, const auto& index)
                          {
                              samurai::static_nested_loop<dim - 1, 0, 2>(
                                  [&](const auto& stencil)
                                  {
                                      auto new_index = 2 * index + stencil;
                                      cl[level + 1][new_index].add_interval(i << 1);
                                  });
                          });
        config.max_level()++;
        mesh = {cl, config};

        u.resize();
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   init_fct(cell);
                               });
        MRadaptation(mra_config);
        jump--;
    }
    samurai::save(std::filesystem::current_path(), fmt::format("initial_mesh_{}_{}_{}", eps, direction, nb), mesh);
    return mesh;
}

void compute_metrics(const auto& mesh, benchmark::State& state)
{
    using mesh_id_t = typename std::decay_t<decltype(mesh)>::mesh_id_t;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();

    int64_t nb_cells     = 0;
    int64_t nb_intervals = 0;
    for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level; ++level)
    {
        auto ghosts_below_cells = samurai::intersection(mesh[mesh_id_t::all_cells][level],
                                                        samurai::union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                      .on(level);

        ghosts_below_cells(
            [&](const auto& i, const auto&)
            {
                nb_cells += 4 * i.size();
                ++nb_intervals;
            });
    }

    double mean = static_cast<double>(nb_cells) / static_cast<double>(nb_intervals);
    double sd   = 0; // standard deviation
    for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level; ++level)
    {
        auto ghosts_below_cells = samurai::intersection(mesh[mesh_id_t::all_cells][level],
                                                        samurai::union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                      .on(level);

        ghosts_below_cells(
            [&](const auto& i, const auto&)
            {
                sd += (static_cast<double>(i.size()) - mean) * (static_cast<double>(i.size()) - mean);
            });
    }
    sd = std::sqrt(sd / static_cast<double>(nb_intervals));

    state.counters["cells"]              = static_cast<double>(nb_cells);
    state.counters["intervals"]          = static_cast<double>(nb_intervals);
    state.counters["cells_per_interval"] = mean;
    state.counters["standard_deviation"] = sd;
}

auto build_mesh_detail(const auto& mesh)
{
    using mesh_t    = std::decay_t<decltype(mesh)>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using cl_type   = typename mesh_t::cl_type;
    using ca_type   = typename mesh_t::ca_type;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();

    cl_type cl;
    for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level; ++level)
    {
        auto ghosts_below_cells = samurai::intersection(mesh[mesh_id_t::all_cells][level],
                                                        samurai::union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                      .on(level);
        ghosts_below_cells(
            [&](const auto& i, const auto& index)
            {
                cl[level][index].add_interval(i);
            });
    }

    return ca_type(cl);
}

template <std::size_t dim>
void benchmark_detail_with_set(benchmark::State& state)
{
    auto mesh       = init_mesh<dim>(1. / static_cast<double>(state.range(0)),
                               static_cast<std::size_t>(state.range(1)),
                               static_cast<std::size_t>(state.range(2)));
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    auto u          = samurai::make_vector_field<double, 1>("u", mesh);
    auto detail     = samurai::make_vector_field<double, 1>("detail", mesh);

    using mesh_t    = decltype(mesh);
    using mesh_id_t = typename mesh_t::mesh_id_t;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();

    compute_metrics(mesh, state);

    for (auto _ : state)
    {
        for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level; ++level)
        {
            auto ghosts_below_cells = samurai::intersection(
                                          mesh[mesh_id_t::all_cells][level],
                                          samurai::union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                          .on(level);
            ghosts_below_cells.apply_op(samurai::compute_detail(detail, u));
        }
    }
    // samurai::save(std::filesystem::current_path(), "mesh_detail_first_version", {true, true}, mesh, detail);

    state.SetItemsProcessed(static_cast<int64_t>(state.counters["cells"]) * state.iterations());
}

template <std::size_t dim>
void benchmark_detail_with_ca(benchmark::State& state)
{
    auto mesh = init_mesh<dim>(1. / static_cast<double>(state.range(0)),
                               static_cast<std::size_t>(state.range(1)),
                               static_cast<std::size_t>(state.range(2)));

    auto u      = samurai::make_scalar_field<double>("u", mesh);
    auto detail = samurai::make_scalar_field<double>("detail", mesh);

    auto mesh_detail = build_mesh_detail(mesh);
    compute_metrics(mesh, state);

    auto op = samurai::compute_detail(detail, u);
    for (auto _ : state)
    {
        samurai::for_each_level(mesh_detail,
                                [&](std::size_t level)
                                {
                                    auto set = samurai::intersection(mesh_detail[level], mesh_detail[level]);
                                    // auto set = samurai::union_(mesh_detail[level], mesh_detail[level]);
                                    set.apply_op(op);
                                });
        // samurai::for_each_interval(mesh_detail,
        //                            [&](std::size_t level, const auto& interval, const auto& index)
        //                            {
        //                                op(level, interval, index);
        //                            });
    }

    int64_t nb_intervals = 0;
    samurai::for_each_level(mesh_detail,
                            [&](std::size_t level)
                            {
                                nb_intervals += mesh_detail[level].shape()[0];
                            });

    state.SetItemsProcessed(4 * static_cast<int64_t>(mesh_detail.nb_cells()) * state.iterations());
    state.counters["cells"]     = static_cast<double>(4 * mesh_detail.nb_cells());
    state.counters["intervals"] = static_cast<double>(nb_intervals);
}

std::vector<std::vector<int64_t>> args_2d = {
    {1000, 10000, 100000},
    {0, 1},
    {1, 2, 3}
};

std::vector<std::vector<int64_t>> args_3d = {
    {1000, 10000},
    {0, 1, 2},
    {1, 2, 3}
};

BENCHMARK(benchmark_detail_with_set<2>)->Unit(benchmark::kMillisecond)->ArgsProduct(args_2d);
BENCHMARK(benchmark_detail_with_ca<2>)->Unit(benchmark::kMillisecond)->ArgsProduct(args_2d);
// BENCHMARK(benchmark_detail_with_set<3>)->Unit(benchmark::kMillisecond)->ArgsProduct(args_3d);
// BENCHMARK(benchmark_detail_with_ca<3>)->Unit(benchmark::kMillisecond)->ArgsProduct(args_3d);
