// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>

#include <xtensor/containers/xfixed.hpp>

#include "samurai_config.hpp"

namespace samurai
{
    namespace args
    {
        // Mesh arguments
        static std::size_t min_level        = std::numeric_limits<std::size_t>::max();
        static std::size_t max_level        = std::numeric_limits<std::size_t>::max();
        static std::size_t start_level      = std::numeric_limits<std::size_t>::max();
        static std::size_t graduation_width = std::numeric_limits<std::size_t>::max();
        static int max_stencil_radius       = std::numeric_limits<int>::max();

        static bool timers = false;
        static bool info   = false;
#ifdef SAMURAI_WITH_MPI
        static bool dont_redirect_output = false;
#endif
        static int finer_level_flux       = 0;
        static bool exact_reconstruction  = false;
        static bool refine_boundary       = false;
        static bool save_debug_fields     = false;
        static bool print_petsc_numbering = false;
        static int sleep_at_startup       = 0;

        static uint32_t load_balancing_at = 0;

        // MRA arguments
        static double epsilon    = std::numeric_limits<double>::infinity();
        static double regularity = std::numeric_limits<double>::infinity();
        static bool rel_detail   = false;
    }

    /**
     * The command-line application returned by samurai::initialize.
     *
     * It is a CLI::App that can also read a point, an xt::xtensor_fixed of shape {N} such as the
     * corners of a Box, from exactly N values: `--max-corner 4 4`. CLI11 does not know this type: it
     * takes it for a wrapper around a single value, so a plain CLI::App keeps the first value only
     * and fills the other coordinates with garbage.
     */
    class App : public CLI::App
    {
      public:

        using CLI::App::add_option;
        using CLI::App::App;

        template <class value_t, std::size_t N, xt::layout_type L, bool sharable>
        CLI::Option*
        add_option(std::string option_name, xt::xtensor_fixed<value_t, xt::xshape<N>, L, sharable>& point, std::string option_description = "")
        {
            auto* option = add_option_function<std::vector<value_t>>(
                std::move(option_name),
                [&point](const std::vector<value_t>& values)
                {
                    std::copy(values.begin(), values.end(), point.begin());
                },
                std::move(option_description));
            option->expected(static_cast<int>(N));
            // Used by capture_default_str(): print the point as CLI11 prints a container.
            option->default_function(
                [&point]()
                {
                    std::ostringstream out;
                    out << '[';
                    for (std::size_t i = 0; i < N; ++i)
                    {
                        out << (i == 0 ? "" : ",") << point[i];
                    }
                    out << ']';
                    return out.str();
                });
            return option;
        }
    };

    SAMURAI_INLINE void read_samurai_arguments(CLI::App& app, int& argc, char**& argv)
    {
        app.add_option("--min-level", args::min_level, "The minimum level of the mesh")->group("SAMURAI");
        app.add_option("--max-level", args::max_level, "The maximum level of the mesh")->group("SAMURAI");
        app.add_option("--start-level", args::start_level, "Start level of AMR")->group("SAMURAI");
        app.add_option("--graduation-width", args::graduation_width, "The graduation width of the mesh")->group("SAMURAI");
        app.add_option("--max-stencil-radius", args::max_stencil_radius, "The maximum number of neighbour in each direction")->group("SAMURAI");
        app.add_option("--load-balancing-at", args::load_balancing_at, "The number of iterations after which load balancing is performed")
            ->group("SAMURAI");

#ifdef SAMURAI_WITH_MPI
        app.add_flag("--dont-redirect-output",
                     args::dont_redirect_output,
                     "Keep the standard output of every rank (by default, only rank 0 prints)")
            ->capture_default_str()
            ->group("IO");
#endif
        app.add_flag("--timers", args::timers, "Print timers at the end of the program")->capture_default_str()->group("Tools");
        app.add_flag("--info", args::info, "Print samurai's version, its dependencies and the build configuration, then exit")->group("Tools");
        app.add_option("--sleep-at-startup",
                       args::sleep_at_startup,
                       "Sleep for a given number of seconds at startup (useful to attach a debugger when running with mpirun/mpiexec)")
            ->capture_default_str()
            ->group("SAMURAI");
        app.add_option(
               "--finer-level-flux",
               args::finer_level_flux,
               "Computation of fluxes at finer levels (default: 0, i.e. no finer level flux, -1 for max_level flux, > 0 for current level + finer_level_flux)")
            ->capture_default_str()
            ->group("SAMURAI");
        app.add_flag("--exact-reconstruction",
                     args::exact_reconstruction,
                     "Reconstruct across refinement boundaries with the real finer cells instead of the flat prediction (LBM stream, FV "
                     "finer-level fluxes, reconstruction()). Removes the projection error at interfaces. Needs a multiresolution mesh, "
                     "rejected on an AMR mesh; default: off")
            ->capture_default_str()
            ->group("SAMURAI");
        app.add_flag("--refine-boundary", args::refine_boundary, "Keep the boundary refined at max_level")->capture_default_str()->group("SAMURAI");
        app.add_flag("--print-petsc-numbering", args::print_petsc_numbering, "Print the local and global numbering used for PETSc")
            ->capture_default_str()
            ->group("SAMURAI");
        app.add_flag("--save-debug-fields", args::save_debug_fields, "Add debug fields during save process (coordinates, indices, levels, ...)")
            ->capture_default_str()
            ->group("SAMURAI");
        app.add_option("--mr-eps", args::epsilon, "The epsilon used by the multiresolution to adapt the mesh")->group("Multiresolution");
        app.add_option("--mr-reg", args::regularity, "The regularity criteria used by the multiresolution to adapt the mesh")
            ->group("Multiresolution");
        app.add_flag("--mr-rel-detail", args::rel_detail, "Use relative detail instead of absolute detail")->group("Multiresolution");
        app.allow_extras();
        app.set_help_flag("", ""); // deactivate --help option
        try
        {
            app.parse(argc, argv);
        }
        catch (const CLI::ParseError& e)
        {
            app.exit(e);
        }
        app.set_help_flag("-h,--help", "Print this help message and exit"); // re-activate --help option
    }
}
