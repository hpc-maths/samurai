// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <CLI/CLI.hpp>

namespace samurai
{
    namespace args
    {
        static bool timers = false;

        static std::size_t disable_batch       = false;
        static std::size_t batch_size          = 100;
        static std::size_t batch_view_min_size = 16;
    }

    inline void read_samurai_arguments(CLI::App& app, int& argc, char**& argv)
    {
        app.add_flag("--timers", args::timers, "Print timers at the end of the program")->capture_default_str()->group("Tools");
        app.add_flag("--disable-batch", args::disable_batch, "Disable batch processing")->capture_default_str()->group("Tools");
        app.add_option("--batch-size", args::batch_size, "Batch size")->capture_default_str()->group("Tools");
        app.add_option("--batch-view-min-size", args::batch_view_min_size, "Min size for a batch by views")->capture_default_str()->group("Tools");
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
