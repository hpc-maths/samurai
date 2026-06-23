// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Comparative load balancing demo on an nD adaptive advection case.
//
// A disk (2D) / sphere (3D) is advected on a multiresolution mesh, so the
// mesh follows the front and the work per process drifts at every adaptation:
// this is the dynamic-imbalance scenario the load balancing strategies are
// made for. Every strategy of the module can be selected on the command line
// and produces the same physics — only the distribution of cells across
// processes differs.
//
// The dimension is fixed at compile time through SAMURAI_LB_DIM, so the same
// source builds both mpi-load-balancing-2d and mpi-load-balancing-3d.
//
//   mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy sfc-hilbert
//   mpiexec -n 4 ./mpi-load-balancing-3d --lb-strategy diffusion --lb-weight level
//   mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy void   # baseline
//
// Useful options:
//   --lb-strategy  void | sfc-morton | sfc-hilbert | diffusion | metis | scotch
//   --lb-weight    uniform | level    (level = 2^(l - min_level): cost of an
//                  explicit scheme with local time stepping)
//   --nt-loadbalance N   rebalance every N steps (default 10)
//   --lb-threshold x     if x > 0, rebalance only when the global imbalance
//                        exceeds x (uses LoadBalancer::required())
//   --lb-dump            write the partition (field "rank") at every
//                        rebalance, for visual comparison in ParaView
//   --lb-stats-file f    append one CSV line per rebalance (rank 0)
//   --lb-skew            (debug) start with every cell on rank 0, to exercise
//                        the strategies on a maximally skewed initial state

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
#include <samurai/load_balancing/strategies/diffusion.hpp>
#include <samurai/load_balancing/strategies/sfc.hpp>
#include <samurai/load_balancing/strategies/void.hpp>
#ifdef SAMURAI_WITH_PARMETIS
#include <samurai/load_balancing/strategies/metis.hpp>
#endif
#ifdef SAMURAI_WITH_PTSCOTCH
#include <samurai/load_balancing/strategies/scotch.hpp>
#endif
#include <samurai/load_balancing/weight.hpp>

#include <filesystem>

#ifndef SAMURAI_LB_DIM
#define SAMURAI_LB_DIM 2
#endif

namespace fs = std::filesystem;
namespace lb = samurai::load_balancing;

namespace
{
    constexpr std::size_t dim = SAMURAI_LB_DIM;

    // Refinement bracket: 3D at max_level 10 would be intractable, so the
    // ceiling is lowered while the base level stays coarse enough for a few
    // processes to share the work.
    constexpr std::size_t demo_min_level = (dim == 2) ? 4 : 3;
    constexpr std::size_t demo_max_level = (dim == 2) ? 10 : 6;

    struct Options
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = xt::zeros<double>({dim});
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = xt::ones<double>({dim});
        std::array<double, dim> velocity                      = []
        {
            std::array<double, dim> v{};
            v.fill(1.);
            return v;
        }();
        double Tf            = 0.1;
        double cfl           = 0.5;
        fs::path path        = fs::current_path();
        std::string filename = "lb_" + std::to_string(dim) + "d";
        std::size_t nfiles   = 1;
        // load balancing
        std::string strategy       = "sfc-hilbert";
        std::string weight         = "uniform";
        std::size_t nt_loadbalance = 10;
        double threshold           = 0.; // 0: rebalance on the period; >0: only when required()
        bool dump_partitions       = false;
        bool skew                  = false;
        std::string stats_file;
        // diffusion strategy options (ignored by the other strategies)
        lb::DiffusionOptions diffusion;
    };

    template <class Field>
    void init(Field& u)
    {
        auto& mesh = u.mesh();
        u.resize();

        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   auto center         = cell.center();
                                   const double radius = .2;
                                   double d2           = 0.;
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       const double c = center[d] - 0.3;
                                       d2 += c * c;
                                   }
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
    int run(const Options& opt, Strategy strategy = {})
    {
        mpi::communicator world;

        // Periodic domain: the advected blob wraps around, so the test is free
        // of boundary-condition complications and the same setup runs in 2D and
        // 3D. (Dirichlet on an adaptive 3D distributed mesh currently trips a
        // core ghost-protocol limitation, see docs/load_balancing_roadmap.md.)
        const samurai::Box<double, dim> box(opt.min_corner, opt.max_corner);
        auto config = samurai::mesh_config<dim>()
                          .min_level(demo_min_level)
                          .max_level(demo_max_level)
                          .max_stencil_size(2)
                          .periodic(true)
                          .disable_minimal_ghost_width();
        auto mesh = samurai::mra::make_mesh(box, config);
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        init(u);

        double t             = 0.;
        double dt            = opt.cfl * mesh.min_cell_length();
        const double dt_save = opt.Tf / static_cast<double>(opt.nfiles);

        auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

        auto MRadaptation = samurai::make_MRAdapt(u);
        auto mra_config   = samurai::mra_config().epsilon(2e-4);
        MRadaptation(mra_config);
        save(opt, u, "_init");

        auto balancer = lb::make_load_balancer<Strategy>(lb::LoadBalanceConfig{.imbalance_threshold = opt.threshold}, std::move(strategy));

        auto weight_is_level = opt.weight == "level";
        auto level_weight    = lb::weight::per_level(
            [&](std::size_t l)
            {
                return std::pow(2.0, static_cast<double>(l) - static_cast<double>(mesh.min_level()));
            });

        // Debug: collapse everything onto rank 0 before the time loop so that
        // the first rebalance starts from a maximally skewed state.
        if (opt.skew)
        {
            balancer.concentrate_on(0, u);
        }

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
                    auto stats = weight_is_level ? balancer.load_balance_with_stats(level_weight, u)
                                                 : balancer.load_balance_with_stats(lb::weight::uniform(), u);
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
    auto& app = samurai::initialize(fmt::format("Compare load balancing strategies on a {}d adaptive advection case", dim), argc, argv);

    Options opt;
    app.add_option("--min-corner", opt.min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", opt.max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", opt.velocity, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", opt.cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", opt.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--lb-strategy", opt.strategy, "Load balancing strategy: void | sfc-morton | sfc-hilbert | diffusion | metis | scotch")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--lb-weight", opt.weight, "Cell weight: uniform | level")->capture_default_str()->group("Load balancing");
    app.add_option("--nt-loadbalance", opt.nt_loadbalance, "Check/rebalance every N time steps")->capture_default_str()->group("Load balancing");
    app.add_option("--lb-threshold", opt.threshold, "If > 0, rebalance only when the global imbalance exceeds this value")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_flag("--lb-dump", opt.dump_partitions, "Dump the partition (field 'rank') at every rebalance")->group("Load balancing");
    app.add_flag("--lb-skew",
                 opt.skew,
                 "Start with every cell on rank 0 (debug: maximally skewed initial state; "
                 "incompatible with metis/scotch, which reject empty subdomains)")
        ->group("Load balancing");
    app.add_option("--lb-stats-file", opt.stats_file, "Append one CSV line per rebalance to this file (rank 0)")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--lb-diffusion-iterations", opt.diffusion.diffusion_iterations, "(diffusion) max iterations of the flux solver")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--lb-diffusion-flux-threshold",
                   opt.diffusion.flux_threshold,
                   "(diffusion) zero out fluxes below this fraction of the mean load")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--lb-diffusion-min-retained",
                   opt.diffusion.min_retained_load_fraction,
                   "(diffusion) fraction of its load a process always keeps")
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

    // one line per strategy: this is the only place to extend when a new
    // strategy is added to the module.
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
    else if (opt.strategy == "diffusion")
    {
        ret = run<lb::Diffusion>(opt, lb::Diffusion{opt.diffusion});
    }
#ifdef SAMURAI_WITH_PARMETIS
    else if (opt.strategy == "metis")
    {
        ret = run<lb::Metis>(opt);
    }
#endif
#ifdef SAMURAI_WITH_PTSCOTCH
    else if (opt.strategy == "scotch")
    {
        ret = run<lb::Scotch>(opt);
    }
#endif
    else
    {
        std::string available = "void | sfc-morton | sfc-hilbert | diffusion";
#ifdef SAMURAI_WITH_PARMETIS
        available += " | metis";
#endif
#ifdef SAMURAI_WITH_PTSCOTCH
        available += " | scotch";
#endif
        if (mpi::communicator{}.rank() == 0)
        {
            std::cerr << "unknown --lb-strategy '" << opt.strategy << "' (expected: " << available << ")" << std::endl;
        }
    }

    samurai::finalize();
    return ret;
}
