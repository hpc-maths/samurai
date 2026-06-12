// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Comparative load balancing demo on a 2D adaptive advection case.
//
// The mesh follows an advected disk (multiresolution), so the work per
// process drifts at every adaptation: this is the dynamic-imbalance scenario
// the load balancing strategies are made for. Every strategy of the module
// can be selected on the command line and produces the same physics — only
// the distribution of cells across processes differs.
//
//   mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy sfc-hilbert
//   mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy sfc-morton --lb-weight level
//   mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy void   # baseline: no balancing
//
// Useful options:
//   --lb-strategy  void | sfc-morton | sfc-hilbert    (steps 4-5 of the
//                  roadmap will add: metis, scotch, diffusion)
//                  KNOWN ISSUE: sfc-hilbert can produce thin subdomain strips
//                  that trigger a pre-existing samurai bug (ghost values not
//                  exchanged beyond 1-cell neighbourhood detection, see
//                  tests/mpi/test_lb_ghosts.cpp) => results may diverge from
//                  the sequential run (observed at np3/np4). Default is
//                  sfc-morton until find_neighbourhood() is fixed.
//   --lb-weight    uniform | level    (level = 2^(l - min_level): cost of an
//                  explicit scheme with local time stepping)
//   --nt-loadbalance N   rebalance every N steps (default 10)
//   --lb-threshold x     if x > 0, rebalance only when the global imbalance
//                        exceeds x (uses LoadBalancer::required())
//   --lb-dump            write the partition (field "rank") at every
//                        rebalance, for visual comparison in ParaView
//   --lb-stats-file f    append one CSV line per rebalance (rank 0)

#include <array>
#include <fstream>
#include <string>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>

#include <samurai/load_balancing/dump.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/sfc.hpp>
#include <samurai/load_balancing/strategies/void.hpp>
#include <samurai/load_balancing/weight.hpp>

#include <filesystem>
namespace fs = std::filesystem;
namespace lb = samurai::load_balancing;

namespace
{
    struct Options
    {
        xt::xtensor_fixed<double, xt::xshape<2>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<2>> max_corner = {1., 1.};
        std::array<double, 2> velocity                      = {1., 1.};
        double Tf                                           = 0.1;
        double cfl                                          = 0.5;
        fs::path path                                       = fs::current_path();
        std::string filename                                = "lb_2d";
        std::size_t nfiles                                  = 1;
        // load balancing
        std::string strategy       = "sfc-morton";
        std::string weight         = "uniform";
        std::size_t nt_loadbalance = 10;
        double threshold           = 0.; // 0: rebalance on the period; >0: only when required()
        bool dump_partitions       = false;
        std::string stats_file;
    };

    template <class Field>
    void init(Field& u)
    {
        auto& mesh = u.mesh();
        u.resize();

        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   auto center           = cell.center();
                                   const double radius   = .2;
                                   const double x_center = 0.3;
                                   const double y_center = 0.3;
                                   const double d2       = (center[0] - x_center) * (center[0] - x_center)
                                                   + (center[1] - y_center) * (center[1] - y_center);
                                   u[cell] = (d2 <= radius * radius) ? 1. : 0.;
                               });
    }

    // note: only u and level here — no per-rank field, so the outputs of runs
    // with different process counts (or strategies) stay comparable with
    // python/compare.py. The partition itself is dumped via --lb-dump.
    template <class Field>
    void save(const Options& opt, const Field& u, const std::string& suffix = "")
    {
        auto& mesh  = u.mesh();
        auto level_ = samurai::make_scalar_field<std::size_t>("level", mesh);

        mpi::communicator world;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   level_[cell] = cell.level;
                               });
        samurai::save(opt.path, fmt::format("{}_size_{}{}", opt.filename, world.size(), suffix), mesh, u, level_);
    }

    /// One CSV line per rebalance, written by rank 0 (header on first call).
    void log_stats(const Options& opt, std::size_t nt, double t, const lb::LoadBalanceStats& stats)
    {
        mpi::communicator world;
        const auto migrated  = boost::mpi::all_reduce(world, stats.cells_migrated_out, std::plus<std::size_t>());
        const auto part_time = boost::mpi::all_reduce(world, stats.partition_time, boost::mpi::maximum<double>());
        const auto migr_time = boost::mpi::all_reduce(world, stats.migration_time, boost::mpi::maximum<double>());

        if (world.rank() == 0)
        {
            std::cout << fmt::format("[lb] ite {}: {} migrated {} cells, imbalance {:.3f} -> {:.3f}",
                                     nt,
                                     stats.strategy_name,
                                     migrated,
                                     stats.imbalance_before,
                                     stats.imbalance_after)
                      << std::endl;

            if (!opt.stats_file.empty())
            {
                const bool write_header = !fs::exists(opt.stats_file);
                std::ofstream csv(opt.stats_file, std::ios::app);
                if (write_header)
                {
                    csv << "nt,t,strategy,weight,ranks,imbalance_before,imbalance_after,"
                           "cells_migrated,partition_time_max,migration_time_max\n";
                }
                csv << fmt::format("{},{},{},{},{},{},{},{},{},{}\n",
                                   nt,
                                   t,
                                   stats.strategy_name,
                                   opt.weight,
                                   world.size(),
                                   stats.imbalance_before,
                                   stats.imbalance_after,
                                   migrated,
                                   part_time,
                                   migr_time);
            }
        }
    }

    /// The whole simulation, generic on the partitioning strategy: this is
    /// the only function a new strategy needs to be plugged into (one line
    /// in the dispatch of main()).
    template <class Strategy>
    int run(const Options& opt)
    {
        constexpr std::size_t dim = 2;
        mpi::communicator world;

        const samurai::Box<double, dim> box(opt.min_corner, opt.max_corner);
        auto config = samurai::mesh_config<dim>().min_level(4).max_level(10).max_stencil_size(2).disable_minimal_ghost_width();
        auto mesh   = samurai::mra::make_mesh(box, config);
        auto u      = samurai::make_scalar_field<double>("u", mesh);
        init(u);
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

        double t             = 0.;
        double dt            = opt.cfl * mesh.min_cell_length();
        const double dt_save = opt.Tf / static_cast<double>(opt.nfiles);

        auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

        auto MRadaptation = samurai::make_MRAdapt(u);
        auto mra_config   = samurai::mra_config().epsilon(2e-4);
        MRadaptation(mra_config);
        save(opt, u, "_init");

        auto balancer = lb::make_load_balancer<Strategy>(lb::LoadBalanceConfig{.imbalance_threshold = opt.threshold});

        auto weight_is_level = opt.weight == "level";
        auto level_weight    = lb::weight::per_level(
            [&](std::size_t l)
            {
                return std::pow(2.0, static_cast<double>(l) - static_cast<double>(mesh.min_level()));
            });

        std::size_t nsave = 1;
        std::size_t nt    = 0;

        while (t != opt.Tf)
        {
            const bool periodic_trigger = ((nt % opt.nt_loadbalance == 0) && nt > 1) || nt == 1;
            if (periodic_trigger)
            {
                // with a threshold, the period only sets how often we *check*
                const bool go = opt.threshold <= 0.
                             || (weight_is_level ? balancer.required(mesh, level_weight) : balancer.required(mesh, lb::weight::uniform()));
                if (go)
                {
                    auto stats = weight_is_level ? balancer.load_balance(level_weight, u) : balancer.load_balance(lb::weight::uniform(), u);
                    log_stats(opt, nt, t, stats);
                    if (opt.dump_partitions)
                    {
                        lb::dump_partition(opt.path, fmt::format("{}_partition_ite_{}", opt.filename, nt), mesh);
                    }
                }
            }

            MRadaptation(mra_config);

            t += dt;
            if (t > opt.Tf)
            {
                dt += opt.Tf - t;
                t = opt.Tf;
            }

            if (world.rank() == 0)
            {
                std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt, t, dt) << std::endl;
            }
            ++nt;

            samurai::update_ghost_mr(u);
            unp1.resize();
            unp1 = u - dt * samurai::upwind(opt.velocity, u);

            std::swap(u.array(), unp1.array());

            if (t >= static_cast<double>(nsave) * dt_save || t == opt.Tf)
            {
                const std::string suffix = (opt.nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
                save(opt, u, suffix);
            }
        }
        return 0;
    }
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Compare load balancing strategies on a 2d adaptive advection case", argc, argv);

    Options opt;
    app.add_option("--min-corner", opt.min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", opt.max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", opt.velocity, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", opt.cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", opt.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--lb-strategy", opt.strategy, "Load balancing strategy: void | sfc-morton | sfc-hilbert")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--lb-weight", opt.weight, "Cell weight: uniform | level")->capture_default_str()->group("Load balancing");
    app.add_option("--nt-loadbalance", opt.nt_loadbalance, "Check/rebalance every N time steps")->capture_default_str()->group("Load balancing");
    app.add_option("--lb-threshold", opt.threshold, "If > 0, rebalance only when the global imbalance exceeds this value")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_flag("--lb-dump", opt.dump_partitions, "Dump the partition (field 'rank') at every rebalance")->group("Load balancing");
    app.add_option("--lb-stats-file", opt.stats_file, "Append one CSV line per rebalance to this file (rank 0)")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--path", opt.path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", opt.filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", opt.nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(opt.path))
    {
        fs::create_directory(opt.path);
    }

    // one line per strategy: this is the only place to extend when the
    // roadmap steps 4-5 add metis, scotch and diffusion
    int ret = 1;
    if (opt.strategy == "void")
    {
        ret = run<lb::Void>(opt);
    }
    else if (opt.strategy == "sfc-morton")
    {
        ret = run<lb::SFC<lb::Morton>>(opt);
    }
    else if (opt.strategy == "sfc-hilbert")
    {
        ret = run<lb::SFC<lb::Hilbert>>(opt);
    }
    else
    {
        if (mpi::communicator{}.rank() == 0)
        {
            std::cerr << "unknown --lb-strategy '" << opt.strategy << "' (expected: void | sfc-morton | sfc-hilbert)" << std::endl;
        }
    }

    samurai::finalize();
    return ret;
}
