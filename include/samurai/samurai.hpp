// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#pragma once

#if defined(SAMURAI_ENABLE_MPI)
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{

    inline void initialize([[maybe_unused]] int& argc, [[maybe_unused]] char**& argv)
    {
#if defined(SAMURAI_ENABLE_MPI)
        MPI_Init(&argc, &argv);
#endif
    }

    inline void finalize()
    {
#if defined(SAMURAI_ENABLE_MPI)
        MPI_Finalize();
#endif
    }

}
