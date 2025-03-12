include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)

if (PETSC_FOUND)
    include_directories(${PETSC_INCLUDE_DIRS})
    find_package(MPI)

    add_executable(multigrid main.cpp)
    target_link_libraries(multigrid samurai CLI11::CLI11 ${PETSC_LINK_LIBRARIES} ${MPI_LIBRARIES})
endif()
