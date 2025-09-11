// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif
#ifdef SAMURAI_WITH_PETSC
#include <petsc.h>
#endif

#include "arguments.hpp"
#include "print.hpp"
#include "timers.hpp"

namespace samurai
{
    inline CLI::App app;

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
        // No output redirection: fmt wrapper handles output uniformly across ranks
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
