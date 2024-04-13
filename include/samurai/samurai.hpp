// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{

    inline void initialize([[maybe_unused]] int& argc, [[maybe_unused]] char**& argv)
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(&argc, &argv);
#endif
    }

    inline void initialize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(nullptr, nullptr);
#endif
    }

    inline void finalize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Finalize();
#endif
    }

}
