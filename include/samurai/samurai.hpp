// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

#include <samurai/timers.hpp>

namespace samurai
{
    namespace times {

        static Timers timers;
    
    }

    inline void initialize([[maybe_unused]] int& argc, [[maybe_unused]] char**& argv)
    {

        int rank     = 0;
        int nproc    = 1;
        int nthreads = 1;
        std::string mpi_tag = "OFF";
        std::string openmp_tag = "OFF";

#ifdef SAMURAI_WITH_MPI
        MPI_Init(&argc, &argv);

        boost::mpi::communicator world;

        rank  = world.rank();
        nproc = world.size();
        mpi_tag = "ON";
#endif

#ifdef SAMURAI_WITH_OPENMP
        openmp_tag = "ON";
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
#endif

        // Message + parallel config
        if( rank == 0 ) {
            std::cout << std::endl;
            std::cout << "     #####     #    #     # #     # ######     #    ### " << std::endl;
            std::cout << "    #     #   # #   ##   ## #     # #     #   # #    #  " << std::endl;
            std::cout << "    #        #   #  # # # # #     # #     #  #   #   #  " << std::endl;
            std::cout << "     #####  #     # #  #  # #     # ######  #     #  #  " << std::endl;
            std::cout << "          # ####### #     # #     # #   #   #######  #  " << std::endl;
            std::cout << "    #     # #     # #     # #     # #    #  #     #  #  " << std::endl;
            std::cout << "     #####  #     # #     #  #####  #     # #     # ### (v 3.14.0)" << std::endl;
            std::cout << std::endl;

            std::cout << fmt::format("Configuration: {} process x {} threads (MPI: {}, OpenMP: {})\n", 
                                     nproc, nthreads, mpi_tag, openmp_tag) << std::endl;

        }

        times::timers.start("smr::total_runtime");
    }

    inline void initialize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(nullptr, nullptr);
#endif
        times::timers.start("smr::total_runtime");
    }

    inline void finalize()
    {
        times::timers.stop("smr::total_runtime");

        times::timers.print();
#ifdef SAMURAI_WITH_MPI
        MPI_Finalize();
#endif
    }

}
