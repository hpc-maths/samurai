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
#ifdef SAMURAI_MEASURE_SET_ALGEBRA
#include "subset/apply.hpp"
#endif
#include "version.hpp"
#include <cstdlib>
#include <thread>

namespace samurai
{
    static CLI::App app;

#ifdef SAMURAI_WITH_PETSC
#define SAMURAI_PARSE(argc, argv)       \
    try                                 \
    {                                   \
        samurai::app.parse(argc, argv); \
        samurai::app.allow_extras();    \
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

    SAMURAI_INLINE auto& initialize(const std::string& description, int& argc, char**& argv)
    {
        app.description(description);
        app.set_config("--config");
        read_samurai_arguments(app, argc, argv);

        if (args::info)
        {
            print_info();
            std::exit(EXIT_SUCCESS);
        }

        std::this_thread::sleep_for(std::chrono::seconds(args::sleep_at_startup));

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
        if (args::timers)
        {
            times::timers.enable();
        }
        times::timers.start("total runtime");

        return app;
    }

    SAMURAI_INLINE auto& initialize(int& argc, char**& argv)
    {
        return initialize("SAMURAI", argc, argv);
    }

    SAMURAI_INLINE void initialize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(nullptr, nullptr);
#endif
    }

    SAMURAI_INLINE void finalize()
    {
        if (args::timers) // cppcheck-suppress knownConditionTrueFalse
        {
            times::timers.stop("total runtime");
            std::cout << std::endl;
            times::timers.print();
#ifdef SAMURAI_MEASURE_SET_ALGEBRA
            const auto& c = detail::set_algebra_counters();
            std::cout << "\nset algebra counters (run totals): applies " << c.applies << ", rows " << c.rows << ", intervals " << c.intervals
                      << ", cells " << c.cells << ", row blocks " << c.blocks << ", empty row segments " << c.empty_segments << std::endl;
            std::cout << "set algebra counters by phase:" << std::endl;
            for (const auto& [phase, pc] : detail::set_algebra_phase_counters())
            {
                std::cout << "  " << phase << ": applies " << pc.applies << ", rows " << pc.rows << ", intervals " << pc.intervals
                          << ", cells " << pc.cells << ", row blocks " << pc.blocks << ", empty row segments " << pc.empty_segments
                          << std::endl;
            }
#endif
        }
#if defined(SAMURAI_WITH_PETSC)
        PetscFinalize();
#elif defined(SAMURAI_WITH_MPI)
        MPI_Finalize();
#endif
    }
}
