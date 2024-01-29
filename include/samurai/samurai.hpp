// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
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
