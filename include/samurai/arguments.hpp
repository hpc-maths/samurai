// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <CLI/CLI.hpp>

namespace samurai
{
    namespace args
    {
        static bool timers                = false;
        static bool enable_max_level_flux = false;
        static bool refine_boundary       = false;
    }

    inline void read_samurai_arguments(CLI::App& app, int& argc, char**& argv)
    {
        app.add_flag("--timers", args::timers, "Print timers at the end of the program")->capture_default_str()->group("Tools");
        app.add_flag("--enable-max-level-flux", args::enable_max_level_flux, "Enable the computation of fluxes at the finest level")
            ->capture_default_str()
            ->group("SAMURAI");
        app.add_flag("--refine-boundary", args::refine_boundary, "Keep the boundary refined at max_level")->capture_default_str()->group("SAMURAI");
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
