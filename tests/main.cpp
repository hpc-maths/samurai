#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif
#ifdef SAMURAI_WITH_PETSC
#include <petsc.h>
#endif
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
#ifdef SAMURAI_WITH_MPI
    boost::mpi::environment env(argc, argv);
#endif
#ifdef SAMURAI_WITH_PETSC
    // PetscInitialize() also initializes MPI, which the PETSc-backed tests need.
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef SAMURAI_WITH_PETSC
    PetscFinalize();
#endif
    return result;
}
