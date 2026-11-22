// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Helper compiled as its own translation unit: it checks that the command-line
// options (`samurai::args`) and the CLI application (`samurai::app`) are shared
// by all the translation units of a program (see test_arguments.cpp).

#include <cstddef>

#include <samurai/arguments.hpp>
#include <samurai/samurai.hpp>

namespace samurai_test
{
    std::size_t max_level_in_other_translation_unit()
    {
        return samurai::args::max_level;
    }

    int finer_level_flux_in_other_translation_unit()
    {
        return samurai::args::finer_level_flux;
    }

    const CLI::App* app_in_other_translation_unit()
    {
        return &samurai::app;
    }
} // namespace samurai_test
