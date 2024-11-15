// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

#include "timers.hpp"

namespace samurai
{

    inline void initialize([[maybe_unused]] int& argc, [[maybe_unused]] char**& argv)
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(&argc, &argv);
#endif
        times::timers.start("total runtime");
    }

    inline void initialize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(nullptr, nullptr);
#endif
        times::timers.start("total runtime");
    }

    inline void finalize()
    {
        times::timers.stop("total runtime");

        times::timers.print();
#ifdef SAMURAI_WITH_MPI
        MPI_Finalize();
#endif
    }

}
