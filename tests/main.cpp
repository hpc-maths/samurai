#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
#ifdef SAMURAI_WITH_MPI
    boost::mpi::environment env(argc, argv);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
