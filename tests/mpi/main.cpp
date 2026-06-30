#include <cstdlib>

#include <boost/mpi.hpp>
#include <gtest/gtest.h>

#include <samurai/arguments.hpp>

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);

    // Allow running the whole MPI test suite through the aggregated
    // (field-merged, non-blocking) ghost-update path, to validate that it is
    // equivalent to the historic one:
    //   SAMURAI_AGGREGATED_GHOST_UPDATE=1 mpirun -n N ./test_xxx
    if (const char* v = std::getenv("SAMURAI_AGGREGATED_GHOST_UPDATE"); v != nullptr && v[0] == '1')
    {
        samurai::args::aggregated_ghost_update = true;
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
