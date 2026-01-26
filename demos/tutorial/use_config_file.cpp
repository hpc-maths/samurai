// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <iostream>

#include <samurai/samurai.hpp>

int main(int argc, char** argv)
{
    samurai::initialize(argc, argv);

    SAMURAI_PARSE(argc, argv);

    if (samurai::args::min_level != 4)
    {
        std::cerr << "Error: Default min-level should be 4." << std::endl;
        return 1;
    }
    if (samurai::args::max_level != 7)
    {
        std::cerr << "Error: Default max-level should be 6." << std::endl;
        return 1;
    }

    samurai::finalize();
    return 0;
}
