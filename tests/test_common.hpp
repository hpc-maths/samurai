#pragma once

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif
#include <functional>

#include <gtest/gtest.h>

#include <xtensor/containers/xarray.hpp>

#include <samurai/interval.hpp>
#include <samurai/list_of_intervals.hpp>

namespace samurai
{
    template <typename coord_t, typename index_t>
    bool operator==(const ListOfIntervals<coord_t, index_t>& li, const xt::xarray<Interval<coord_t, index_t>>& array)
    {
        auto ix = li.cbegin();
        auto iy = array.cbegin();
        while (ix != li.cend() && iy != array.cend())
        {
            if (*ix != *iy)
            {
                return false;
            }
            ++ix;
            ++iy;
        }
        if (ix == li.cend() && iy == array.cend())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // test_samurai_lib also runs under mpiexec in an MPI build. A mesh is then
    // split across the processes, so a count or an integral over the local cells
    // is only a part of the whole.
    namespace test
    {
        // Number of processes the test runs on (1 in a sequential build).
        inline int nb_processes()
        {
#ifdef SAMURAI_WITH_MPI
            return boost::mpi::communicator().size();
#else
            return 1;
#endif
        }

        // True on the process that prints the reports gathered from all processes.
        inline bool is_root_process()
        {
#ifdef SAMURAI_WITH_MPI
            return boost::mpi::communicator().rank() == 0;
#else
            return true;
#endif
        }

        // Sum over all processes of a quantity computed on the local cells.
        // Collective: every process must call it.
        template <class T>
        T global_sum(const T& local)
        {
#ifdef SAMURAI_WITH_MPI
            return boost::mpi::all_reduce(boost::mpi::communicator(), local, std::plus<T>());
#else
            return local;
#endif
        }

        // Minimum over all processes. Collective: every process must call it.
        template <class T>
        T global_min(const T& local)
        {
#ifdef SAMURAI_WITH_MPI
            return boost::mpi::all_reduce(boost::mpi::communicator(), local, boost::mpi::minimum<T>());
#else
            return local;
#endif
        }

        // Maximum over all processes. Collective: every process must call it.
        template <class T>
        T global_max(const T& local)
        {
#ifdef SAMURAI_WITH_MPI
            return boost::mpi::all_reduce(boost::mpi::communicator(), local, boost::mpi::maximum<T>());
#else
            return local;
#endif
        }
    }
}

// Skips a test whose assertions only hold when a single process holds the whole
// mesh: a given cell, a mesh built from a cell list given on every process, ...
#define SKIP_IF_DISTRIBUTED(reason)                  \
    if (::samurai::test::nb_processes() > 1)         \
    {                                                \
        GTEST_SKIP() << "distributed mesh: " reason; \
    }
