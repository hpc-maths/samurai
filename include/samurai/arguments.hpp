// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <CLI/CLI.hpp>

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

        inline bool timers = false;
#ifdef SAMURAI_WITH_MPI
        inline bool print_root_only = false;
#endif
        inline int finer_level_flux   = 0;
        inline bool refine_boundary   = false;
        inline bool save_debug_fields = false;

        // MRA arguments
        inline double epsilon    = std::numeric_limits<double>::infinity();
        inline double regularity = std::numeric_limits<double>::infinity();
        inline bool rel_detail   = false;
    }

    inline void read_samurai_arguments(CLI::App& app, int& argc, char**& argv)
    {
        app.add_option("--min-level", args::min_level, "The minimum level of the mesh")->group("SAMURAI");
        app.add_option("--max-level", args::max_level, "The maximum level of the mesh")->group("SAMURAI");
        app.add_option("--start-level", args::start_level, "Start level of AMR")->group("SAMURAI");
        app.add_option("--graduation-width", args::graduation_width, "The graduation width of the mesh")->group("SAMURAI");
        app.add_option("--max-stencil-radius", args::max_stencil_radius, "The maximum number of neighbour in each direction")->group("SAMURAI");

#ifdef SAMURAI_WITH_MPI
        app.add_flag("--print-root-only", args::print_root_only, "Print on root rank only by default for samurai::io::print/eprint")
            ->capture_default_str()
            ->group("IO");
#endif
        app.add_flag("--timers", args::timers, "Print timers at the end of the program")->capture_default_str()->group("Tools");
        app.add_option(
               "--finer-level-flux",
               args::finer_level_flux,
               "Computation of fluxes at finer levels (default: 0, i.e. no finer level flux, -1 for max_level flux, > 0 for current level + finer_level_flux)")
            ->capture_default_str()
            ->group("SAMURAI");
        app.add_flag("--refine-boundary", args::refine_boundary, "Keep the boundary refined at max_level")->capture_default_str()->group("SAMURAI");
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
