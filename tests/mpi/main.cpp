#include <boost/mpi.hpp>
#ifdef SAMURAI_WITH_PETSC
#include <petsc.h>
#endif
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
#ifdef SAMURAI_WITH_PETSC
    // MPI is already initialized: PetscInitialize() reuses it and PetscFinalize() leaves it to boost.
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef SAMURAI_WITH_PETSC
    PetscFinalize();
#endif
    return result;
}
