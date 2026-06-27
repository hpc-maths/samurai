// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Interactive load balancing demo on a thin "tube" domain [0, width] x [0,
// length] (default 1 x 10). A configurable number of circles is scattered
// along the tube; the mesh is refined (multiresolution) around their
// interfaces, then a load balancing strategy of your choice partitions the
// cells across the MPI processes. The partition is written to disk (field
// "rank") so the strategies can be compared visually in ParaView.
//
// This is the static counterpart of mpi-load-balancing-*d (which advects a
// blob): here you play with the geometry (number/size of circles, tube aspect
// ratio) and the strategy, and look at how each strategy cuts the domain.
//
//   mpiexec -n 4 ./mpi-load-balancing-tube --ncircles 6 --lb-strategy sfc-hilbert
//   mpiexec -n 4 ./mpi-load-balancing-tube --ncircles 6 --lb-strategy diffusion
//   mpiexec -n 4 ./mpi-load-balancing-tube --ncircles 3 --length 20 --lb-strategy sfc-morton
//
// Useful options:
//   --ncircles N    number of circles scattered along the tube (default 4)
//   --radius r      circle radius (default 0.15)
//   --length L      tube length along y (width is fixed to 1; default 10)
//   --lb-strategy   void | sfc-morton | sfc-hilbert | diffusion | metis | scotch
//   --lb-weight     uniform | level   (level = 2^(l - min_level))
//   --lb-iterations N   number of balancing passes (default 1; raised to 30 for
//                       diffusion, which is incremental). Stops early once the
//                       imbalance falls below 2%.
//   --min-level / --max-level   refinement bracket (default 4 .. 9)

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

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

namespace fs  = std::filesystem;
namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    constexpr std::size_t dim = 2; // a tube is intrinsically 2D here

    struct Options
    {
        // geometry
        std::size_t ncircles = 4;
        double radius        = 0.15;
        double length        = 10.; // tube along y; width is 1 along x
        // refinement
        std::size_t min_level = 4;
        std::size_t max_level = 9;
        double epsilon        = 1e-3;
        // load balancing
        std::string strategy   = "sfc-hilbert";
        std::string weight     = "uniform";
        std::size_t iterations = 1;
        bool skew              = false;
        // output
        fs::path path        = fs::current_path();
        std::string filename = "tube";
        // diffusion options (ignored by the other strategies)
        lb::DiffusionOptions diffusion;
    };

    /// Circle centers/radii scattered along the tube: evenly spaced in y,
    /// alternating left/right of the center line in x. A clear, reproducible
    /// layout that exercises the 2D locality of the partitioners.
    std::vector<std::array<double, 3>> make_circles(const Options& opt)
    {
        std::vector<std::array<double, 3>> circles;
        circles.reserve(opt.ncircles);
        for (std::size_t k = 0; k < opt.ncircles; ++k)
        {
            const double y    = opt.length * (static_cast<double>(k) + 0.5) / static_cast<double>(opt.ncircles);
            const double side = (k % 2 == 0) ? -1. : 1.;
            const double x    = 0.5 + 0.25 * side; // 0.25 or 0.75 of the unit width
            circles.push_back({x, y, opt.radius});
        }
        return circles;
    }

    /// Filled-disk indicator (1 inside any circle, 0 outside): the 0->1 jump at
    /// each interface is what the multiresolution adaptation refines.
    template <class Field>
    void init(const Options& opt, Field& u)
    {
        auto& mesh         = u.mesh();
        const auto circles = make_circles(opt);
        u.resize();
        u.fill(0.);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   const auto center = cell.center();
                                   for (const auto& c : circles)
                                   {
                                       const double dx = center[0] - c[0];
                                       const double dy = center[1] - c[1];
                                       if (dx * dx + dy * dy <= c[2] * c[2])
                                       {
                                           u[cell] = 1.;
                                           return;
                                       }
                                   }
                               });
    }

    /// Save the mesh with the owning rank, the level and the circle field, so
    /// the partition can be inspected (color by "rank") in ParaView.
    template <class Field>
    void save_state(const Options& opt, Field& u, const std::string& suffix)
    {
        auto& mesh = u.mesh();
        mpi::communicator world;

        auto rank_  = samurai::make_scalar_field<int>("rank", mesh);
        auto level_ = samurai::make_scalar_field<std::size_t>("level", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   rank_[cell]  = world.rank();
                                   level_[cell] = cell.level;
                               });
        const std::string name = fmt::format("{}_{}_nc{}_size_{}{}", opt.filename, opt.strategy, opt.ncircles, world.size(), suffix);
        samurai::save(opt.path, name, mesh, u, rank_, level_);
    }

    void log_stats(std::size_t pass, const lb::LoadBalanceStats& stats)
    {
        mpi::communicator world;
        const auto migrated = mpi::all_reduce(world, stats.cells_migrated_out, std::plus<std::size_t>());
        if (world.rank() == 0)
        {
            std::cout << fmt::format("[lb] pass {}: {} migrated {} cells, imbalance {:.3f} -> {:.3f}",
                                     pass,
                                     stats.strategy_name,
                                     migrated,
                                     stats.imbalance_before,
                                     stats.imbalance_after)
                      << std::endl;
        }
    }

    /// Helper "strategy" that imposes a connected but heavily imbalanced
    /// decomposition: rank 0 owns the bottom 70% of the tube, the other ranks
    /// share the top in contiguous y-slabs. Unlike collapsing everything onto
    /// rank 0 (which removes every inter-rank interface and leaves the *local*
    /// diffusion strategy nothing to move), this keeps the ranks adjacent, so
    /// every strategy -- global or local -- visibly rebalances from it.
    struct BandedImbalance
    {
        double length;

        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& /*weight*/) const
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            mpi::communicator world;
            const int size = world.size();
            auto flags     = samurai::make_scalar_field<int>("lb_flags", mesh);
            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](const auto& cell)
                                   {
                                       const double y  = cell.center(1);
                                       const double y0 = 0.7 * length; // heavy slab [0, y0) -> rank 0
                                       int r           = 0;
                                       if (y >= y0 && size > 1)
                                       {
                                           const double frac = (y - y0) / (length - y0);
                                           r                 = 1 + static_cast<int>(frac * static_cast<double>(size - 1));
                                           r                 = (r >= size) ? size - 1 : r;
                                       }
                                       flags[cell] = r;
                                   });
            return flags;
        }

        std::string name() const
        {
            return "banded-init";
        }
    };

    /// Generic on the partitioning strategy: build the tube, refine around the
    /// circles, then balance and dump the partition. Adding a strategy is one
    /// line in the dispatch of main().
    template <class Strategy>
    int run(const Options& opt, Strategy strategy = {})
    {
        mpi::communicator world;

        // Periodic tube: avoids boundary-condition handling on the adaptive
        // distributed mesh (same choice as the advection demo).
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., opt.length};
        const samurai::Box<double, dim> box(min_corner, max_corner);

        auto config = samurai::mesh_config<dim>()
                          .min_level(opt.min_level)
                          .max_level(opt.max_level)
                          .max_stencil_size(2)
                          .periodic(true)
                          .disable_minimal_ghost_width();
        auto mesh = samurai::mra::make_mesh(box, config);
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        init(opt, u);

        // refine around the circle interfaces
        auto MRadaptation = samurai::make_MRAdapt(u);
        auto mra_config   = samurai::mra_config().epsilon(opt.epsilon);
        MRadaptation(mra_config);
        init(opt, u); // crisp indicator on the adapted mesh for display

        // Optionally start from a heavy, imbalanced (but connected) decomposition
        // so the chosen strategy visibly rebalances -- including diffusion, which
        // needs inter-rank interfaces to move cells (see BandedImbalance).
        if (opt.skew)
        {
            lb::make_load_balancer<BandedImbalance>(lb::LoadBalanceConfig{}, BandedImbalance{opt.length}).load_balance(lb::weight::uniform(), u);
        }

        save_state(opt, u, "_init"); // decomposition before balancing

        auto balancer = lb::make_load_balancer<Strategy>(lb::LoadBalanceConfig{}, std::move(strategy));

        const bool weight_is_level = opt.weight == "level";
        auto level_weight          = lb::weight::per_level(
            [&](std::size_t l)
            {
                return std::pow(2.0, static_cast<double>(l) - static_cast<double>(mesh.min_level()));
            });

        constexpr double stop_bound = 0.05;
        for (std::size_t pass = 0; pass < opt.iterations; ++pass)
        {
            auto stats = weight_is_level ? balancer.load_balance_with_stats(level_weight, u)
                                         : balancer.load_balance_with_stats(lb::weight::uniform(), u);
            log_stats(pass, stats);
            if (stats.imbalance_after <= stop_bound)
            {
                break;
            }
        }

        save_state(opt, u, "_balanced"); // the strategy's partition

        const auto total = mpi::all_reduce(world, mesh.nb_cells(), std::plus<std::size_t>());
        if (world.rank() == 0)
        {
            std::cout << fmt::format("[lb] {} circles, {} cells, {} ranks, strategy '{}', weight '{}'",
                                     opt.ncircles,
                                     total,
                                     world.size(),
                                     opt.strategy,
                                     opt.weight)
                      << std::endl;
        }
        return 0;
    }
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Load balancing strategies on a tube with a configurable number of circles", argc, argv);

    Options opt;
    app.add_option("--ncircles", opt.ncircles, "Number of circles scattered along the tube")->capture_default_str()->group("Geometry");
    app.add_option("--radius", opt.radius, "Circle radius")->capture_default_str()->group("Geometry");
    app.add_option("--length", opt.length, "Tube length along y (width is 1)")->capture_default_str()->group("Geometry");
    // note: --min-level / --max-level / --mr-eps are registered by samurai
    // itself and flow into the mesh / adaptation config (see make_mesh).
    app.add_option("--mr-epsilon", opt.epsilon, "Multiresolution threshold")->capture_default_str()->group("Refinement");
    app.add_option("--lb-strategy", opt.strategy, "Load balancing strategy: void | sfc-morton | sfc-hilbert | diffusion | metis | scotch")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--lb-weight", opt.weight, "Cell weight: uniform | level")->capture_default_str()->group("Load balancing");
    app.add_option("--lb-iterations", opt.iterations, "Number of balancing passes (raised to 30 for diffusion)")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_flag("--skew", opt.skew, "Start from a heavy, imbalanced (but connected) decomposition so the strategy visibly rebalances")
        ->group("Load balancing");
    app.add_option("--lb-diffusion-iterations", opt.diffusion.diffusion_iterations, "(diffusion) max iterations of the flux solver")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--lb-diffusion-relaxation",
                   opt.diffusion.relaxation,
                   "(diffusion) under-relaxation factor in (0,1]: shed only this fraction of the flux per pass "
                   "to damp the oscillation that appears with many ranks (try 0.5)")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_flag("--lb-diffusion-monotonic,!--no-lb-diffusion-monotonic",
                 opt.diffusion.monotonic,
                 "(diffusion) cap each cession at half the local load gradient (deal-agreement): forbids the "
                 "receiver from passing the giver, so the imbalance decreases monotonically and cannot enter a "
                 "limit cycle. On by default; pass --no-lb-diffusion-monotonic to compare")
        ->capture_default_str()
        ->group("Load balancing");
    app.add_option("--path", opt.path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", opt.filename, "File name prefix")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    if (opt.ncircles == 0)
    {
        opt.ncircles = 1;
    }
    // diffusion is incremental: give it several passes unless the user chose one
    if (opt.strategy == "diffusion" && opt.iterations == 1)
    {
        opt.iterations = 30;
    }

    if (!fs::exists(opt.path))
    {
        fs::create_directory(opt.path);
    }

    // one line per strategy: the only place to extend for a new strategy.
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
