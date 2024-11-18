// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

#include "arguments.hpp"
#include "timers.hpp"

namespace samurai
{
    inline void initialize(CLI::App& app, int& argc, char**& argv)
    {
        read_samurai_arguments(app, argc, argv);

#ifdef SAMURAI_WITH_MPI
        MPI_Init(&argc, &argv);
#endif
        times::timers.start("total runtime");
    }

    inline void initialize(int& argc, char**& argv)
    {
        CLI::App app{"SAMURAI"};
        initialize(app, argc, argv);
    }

    inline void initialize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(nullptr, nullptr);
#endif
    }

    inline void finalize()
    {
        if (args::timers)
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
