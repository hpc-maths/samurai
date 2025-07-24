// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <fstream>
namespace mpi = boost::mpi;
#endif

#include "arguments.hpp"
#include "timers.hpp"

namespace samurai
{
    static CLI::App app;

#define SAMURAI_PARSE(argc, argv)       \
    try                                 \
    {                                   \
        samurai::app.parse(argc, argv); \
    }                                   \
    catch (const CLI::ParseError& e)    \
    {                                   \
        return samurai::app.exit(e);    \
    }

#ifdef SAMURAI_WITH_PETSC
    void petsc_initialize(int& argc, char**& argv)
    {
        samurai::times::timers.start("petsc init");
        PetscInitialize(&argc, &argv, 0, nullptr);

        // If on, Petsc will issue warnings saying that the options managed by CLI are unused
        PetscOptionsSetValue(NULL, "-options_left", "off");
        samurai::times::timers.stop("petsc init");
    }

    void petsc_finalize()
    {
        samurai::times::timers.start("petsc finalize");
        PetscFinalize();
        samurai::times::timers.stop("petsc finalize");
    }
#endif

    inline auto& initialize(const std::string& description, int& argc, char**& argv)
    {
        app.description(description);
        read_samurai_arguments(app, argc, argv);

#ifdef SAMURAI_WITH_MPI
        MPI_Init(&argc, &argv);
        // redirect stdout to /dev/null for all ranks except rank 0
        mpi::communicator world;
        if (!args::dont_redirect_output && world.rank() != 0) // cppcheck-suppress knownConditionTrueFalse
        {
            static std::ofstream null_stream("/dev/null");
            std::cout.rdbuf(null_stream.rdbuf());
        }
#endif
        times::timers.start("total runtime");

#ifdef SAMURAI_WITH_PETSC
        petsc_initialize(argc, argv);
#endif
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
#ifdef SAMURAI_WITH_PETSC
        petsc_finalize();
#endif
        if (args::timers) // cppcheck-suppress knownConditionTrueFalse
        {
            times::timers.stop("total runtime");
            std::cout << std::endl;
            times::timers.print();
        }
#ifdef SAMURAI_WITH_MPI
        MPI_Finalize();
#endif
    }

}
