// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <fstream>
namespace mpi = boost::mpi;
#endif
#ifdef SAMURAI_WITH_PETSC
#include <petsc.h>
#endif

#include "arguments.hpp"
#include "timers.hpp"

namespace samurai
{
    static CLI::App app;

#ifdef SAMURAI_WITH_PETSC
#define SAMURAI_PARSE(argc, argv)       \
    try                                 \
    {                                   \
        samurai::app.parse(argc, argv); \
        app.allow_extras();             \
    }                                   \
    catch (const CLI::ParseError& e)    \
    {                                   \
        return samurai::app.exit(e);    \
    }
#else
#define SAMURAI_PARSE(argc, argv)       \
    try                                 \
    {                                   \
        samurai::app.parse(argc, argv); \
    }                                   \
    catch (const CLI::ParseError& e)    \
    {                                   \
        return samurai::app.exit(e);    \
    }
#endif

    inline auto& initialize(const std::string& description, int& argc, char**& argv)
    {
        app.description(description);
        read_samurai_arguments(app, argc, argv);

        // sleep(8);

#if defined(SAMURAI_WITH_PETSC)
        // MPI_Init() in called by PetscInitialize()
        PetscInitialize(&argc, &argv, 0, nullptr);
        // If on, Petsc will issue warnings saying that the options managed by CLI are unused
        PetscOptionsSetValue(NULL, "-options_left", "off");
#elif defined(SAMURAI_WITH_MPI)
        MPI_Init(&argc, &argv);
#endif

#if defined(SAMURAI_WITH_MPI)
        // redirect stdout to /dev/null for all ranks except rank 0
        mpi::communicator world;
        if (!args::dont_redirect_output && world.rank() != 0) // cppcheck-suppress knownConditionTrueFalse
        {
            static std::ofstream null_stream("/dev/null");
            std::cout.rdbuf(null_stream.rdbuf());
        }
#endif
        times::timers.start("total runtime");

        return app;
    }

    inline auto& initialize(int& argc, char**& argv)
    {
        return initialize("SAMURAI", argc, argv);
    }

    inline void initialize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(nullptr, nullptr);
#endif
    }

    inline void finalize()
    {
        if (args::timers) // cppcheck-suppress knownConditionTrueFalse
        {
            times::timers.stop("total runtime");
            std::cout << std::endl;
            times::timers.print();
        }
#if defined(SAMURAI_WITH_PETSC)
        PetscFinalize();
#elif defined(SAMURAI_WITH_MPI)
        MPI_Finalize();
#endif
    }
}
