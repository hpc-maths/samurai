// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Micro-benchmarks for the load balancing module.
//
// Two costs are measured per strategy on a uniform distributed mesh:
//   * BM_partition  — the strategy's partition() call alone (the part that
//                     differs between strategies: SFC sort+scan, diffusion
//                     flux solve, graph build + ParMETIS/PT-Scotch);
//   * BM_balance    — a full load_balance() pass starting from a maximally
//                     skewed state (everything on rank 0), i.e. partition +
//                     migration + mesh rebuild. The skew is re-applied in a
//                     paused region so only the rebalancing is timed.
//
// These objectivise "the rebalancing cost stays negligible against a time
// step": compare the reported times to the cost of one explicit update on
// the same mesh.
//
// The benchmark is only meaningful under mpiexec with more than one process
// (with a single process partition() is trivial and load_balance() a no-op).
// A FIXED iteration count is used so that every rank runs the exact same
// number of collective operations and cannot desynchronise.

#include <cstddef>
#include <string>

#include <benchmark/benchmark.h>

#include <boost/mpi.hpp>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/diffusion.hpp>
#include <samurai/load_balancing/strategies/sfc.hpp>
#include <samurai/load_balancing/strategies/void.hpp>
#include <samurai/load_balancing/weight.hpp>
#ifdef SAMURAI_WITH_PARMETIS
#include <samurai/load_balancing/strategies/metis.hpp>
#endif
#ifdef SAMURAI_WITH_PTSCOTCH
#include <samurai/load_balancing/strategies/scotch.hpp>
#endif

namespace lb = samurai::load_balancing;

namespace
{
    // Number of timed repetitions; fixed so all ranks stay in lockstep.
    constexpr int bench_iterations = 10;

    // Uniform mesh levels (kept modest: BM_balance migrates the whole mesh
    // every iteration, so a large mesh makes the run very long).
    constexpr int level_2d = 7; // 128 x 128
    constexpr int level_3d = 4; //  16^3

    template <std::size_t dim>
    auto make_uniform_mesh(std::size_t level)
    {
        const xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = xt::zeros<double>({dim});
        const xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = xt::ones<double>({dim});
        const samurai::Box<double, dim> box(min_corner, max_corner);
        auto config = samurai::mesh_config<dim>().min_level(level).max_level(level).periodic(true);
        return samurai::mra::make_mesh(box, config);
    }

    /// Cost of partition() alone (read-only, no migration).
    template <std::size_t dim, class Strategy>
    void BM_partition(benchmark::State& state)
    {
        auto mesh = make_uniform_mesh<dim>(static_cast<std::size_t>(state.range(0)));
        Strategy strategy;
        auto weight = lb::weight::uniform();

        using mesh_id_t = typename decltype(mesh)::mesh_id_t;
        for (auto _ : state)
        {
            auto flags = strategy.partition(mesh, weight);
            benchmark::DoNotOptimize(flags.array());
        }
        state.counters["cells/rank"] = static_cast<double>(mesh.nb_cells(mesh_id_t::cells));
    }

    /// Cost of a full rebalancing pass from a fully skewed state.
    template <std::size_t dim, class Strategy>
    void BM_balance(benchmark::State& state)
    {
        auto mesh = make_uniform_mesh<dim>(static_cast<std::size_t>(state.range(0)));
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        u.fill(1.);
        auto balancer = lb::make_load_balancer<Strategy>();
        auto weight   = lb::weight::uniform();

        for (auto _ : state)
        {
            state.PauseTiming();
            balancer.concentrate_on(0, u); // skew: everything onto rank 0
            state.ResumeTiming();
            auto stats = balancer.load_balance_with_stats(weight, u);
            benchmark::DoNotOptimize(stats.cells_migrated_in);
        }
    }

    // partition() is the part that differs between strategies, so it is
    // benchmarked for every one of them, on a 2D and a 3D mesh.
    template <class Strategy>
    void register_partition(const std::string& name)
    {
        benchmark::RegisterBenchmark((name + "/partition/2d").c_str(), BM_partition<2, Strategy>)
            ->Arg(level_2d)
            ->Iterations(bench_iterations)
            ->Unit(benchmark::kMicrosecond)
            ->UseRealTime();
        benchmark::RegisterBenchmark((name + "/partition/3d").c_str(), BM_partition<3, Strategy>)
            ->Arg(level_3d)
            ->Iterations(bench_iterations)
            ->Unit(benchmark::kMicrosecond)
            ->UseRealTime();
    }

    // The migration + rebuild cost is strategy-independent (the same migrate()
    // path runs whatever produced the flags), so a full load_balance() pass is
    // benchmarked only for strategies that tolerate the empty subdomains the
    // maximal skew creates. ParMETIS / PT-Scotch reject an empty local graph,
    // so they are excluded here; their partition() cost is captured above.
    template <class Strategy>
    void register_balance(const std::string& name)
    {
        benchmark::RegisterBenchmark((name + "/balance/2d").c_str(), BM_balance<2, Strategy>)
            ->Arg(level_2d)
            ->Iterations(bench_iterations)
            ->Unit(benchmark::kMicrosecond)
            ->UseRealTime();
        benchmark::RegisterBenchmark((name + "/balance/3d").c_str(), BM_balance<3, Strategy>)
            ->Arg(level_3d)
            ->Iterations(bench_iterations)
            ->Unit(benchmark::kMicrosecond)
            ->UseRealTime();
    }

    /// Reporter that prints nothing: used on every rank but rank 0.
    class NullReporter : public benchmark::BenchmarkReporter
    {
      public:

        bool ReportContext(const Context&) override
        {
            return true;
        }

        void ReportRuns(const std::vector<Run>&) override
        {
        }
    };
}

int main(int argc, char** argv)
{
    samurai::initialize(argc, argv);
    benchmark::Initialize(&argc, argv);

    register_partition<lb::SFC<lb::Morton>>("sfc-morton");
    register_partition<lb::SFC<lb::Hilbert>>("sfc-hilbert");
    register_partition<lb::Diffusion>("diffusion");
#ifdef SAMURAI_WITH_PARMETIS
    register_partition<lb::Metis>("metis");
#endif
#ifdef SAMURAI_WITH_PTSCOTCH
    register_partition<lb::Scotch>("scotch");
#endif

    // Full-pass (partition + migration + rebuild) only for strategies that
    // accept empty subdomains; see register_balance().
    register_balance<lb::SFC<lb::Morton>>("sfc-morton");
    register_balance<lb::SFC<lb::Hilbert>>("sfc-hilbert");
    register_balance<lb::Diffusion>("diffusion");

    boost::mpi::communicator world;
    if (world.rank() == 0)
    {
        benchmark::RunSpecifiedBenchmarks();
    }
    else
    {
        NullReporter null;
        benchmark::RunSpecifiedBenchmarks(&null);
    }
    benchmark::Shutdown();

    samurai::finalize();
    return 0;
}
