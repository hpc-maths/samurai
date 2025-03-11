include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)
if (PETSC_FOUND)
    include_directories(${PETSC_INCLUDE_DIRS})
    find_package(MPI)

    add_executable(highorder main.cpp)

    target_link_libraries(highorder samurai CLI11::CLI11 ${PETSC_LINK_LIBRARIES} ${MPI_LIBRARIES})
endif()
