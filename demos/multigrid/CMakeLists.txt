include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)
find_package(MPI)

add_executable(multigrid main.cpp)

target_link_libraries(multigrid samurai ${PETSC_LIBRARIES} ${MPI_LIBRARIES})