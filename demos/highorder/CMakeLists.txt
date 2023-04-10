include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)
if (PETSC_FOUND)
    find_package(MPI)

    add_executable(highorder main.cpp)

    target_link_libraries(highorder samurai ${PETSC_LIBRARIES} ${MPI_LIBRARIES})
endif()
