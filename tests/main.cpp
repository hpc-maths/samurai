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

    int ret = RUN_ALL_TESTS();
}
