// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <CLI/CLI.hpp>

namespace samurai
{
    namespace args
    {
        static bool timers = false;
    }

    auto read_samurai_arguments(CLI::App& app, int& argc, char**& argv)
    {
        app.add_flag("--timers", args::timers, "Print timers at the end of the program")->capture_default_str()->group("SAMURAI tools");
        app.allow_extras();
        app.set_help_flag("", "");
        try
        {
            app.parse(argc, argv);
        }
        catch (const CLI::ParseError& e)
        {
            app.exit(e);
        }
        app.set_help_flag("-h,--help", "Print this help message and exit");
    }
}
