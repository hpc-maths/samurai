include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)

if(PETSC_FOUND)
    find_package(MPI)

    add_executable(finite-volume-heat heat.cpp)
    target_link_libraries(finite-volume-heat samurai CLI11::CLI11 ${PETSC_LIBRARIES} ${MPI_LIBRARIES})

    add_executable(finite-volume-stokes-2d stokes_2d.cpp)
    target_link_libraries(finite-volume-stokes-2d samurai CLI11::CLI11 ${PETSC_LIBRARIES} ${MPI_LIBRARIES})
endif()

add_executable(finite-volume-amr-burgers-hat AMR_Burgers_Hat.cpp)
target_link_libraries(finite-volume-amr-burgers-hat samurai CLI11::CLI11)

add_executable(finite-volume-level-set level_set.cpp)
target_link_libraries(finite-volume-level-set samurai CLI11::CLI11)

add_executable(finite-volume-level-set-from-scratch level_set_from_scratch.cpp)
target_link_libraries(finite-volume-level-set-from-scratch samurai CLI11::CLI11)

if(MSVC)
    target_compile_options(finite-volume-level-set-from-scratch PUBLIC /bigobj)
    target_compile_options(finite-volume-level-set PUBLIC /bigobj)
endif()

add_executable(finite-volume-advection-1d advection_1d.cpp)
target_link_libraries(finite-volume-advection-1d samurai CLI11::CLI11)

add_executable(finite-volume-advection-2d advection_2d.cpp)
target_link_libraries(finite-volume-advection-2d samurai CLI11::CLI11)

add_executable(finite-volume-advection-2d-user-bc advection_2d_user_bc.cpp)
target_link_libraries(finite-volume-advection-2d-user-bc samurai CLI11::CLI11)

add_executable(finite-volume-scalar-burgers-2d scalar_burgers_2d.cpp)
target_link_libraries(finite-volume-scalar-burgers-2d samurai CLI11::CLI11)

add_executable(finite-volume-burgers burgers.cpp)
target_link_libraries(finite-volume-burgers samurai CLI11::CLI11)

# add_subdirectory(BZ)
