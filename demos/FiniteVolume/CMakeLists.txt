# List of files requiring PETSc
set(PETSC_DEMOS
heat.cpp:finite-volume-heat
heat_heterogeneous.cpp:finite-volume-heat-heterogeneous
heat_nonlinear.cpp:finite-volume-heat-nonlinear
stokes_2d.cpp:finite-volume-stokes-2d
lid_driven_cavity.cpp:finite-volume-lid-driven-cavity
nagumo.cpp:finite-volume-nagumo
manual_block_matrix_assembly.cpp:manual_block_matrix_assembly
)

# List of standard files (without PETSc)
set(STANDARD_DEMOS
AMR_Burgers_Hat.cpp:finite-volume-amr-burgers-hat
level_set.cpp:finite-volume-level-set
level_set_from_scratch.cpp:finite-volume-level-set-from-scratch
advection_1d.cpp:finite-volume-advection-1d
advection_2d.cpp:finite-volume-advection-2d
advection_2d_user_bc.cpp:finite-volume-advection-2d-user-bc
scalar_burgers_2d.cpp:finite-volume-scalar-burgers-2d
linear_convection.cpp:finite-volume-linear-convection
burgers.cpp:finite-volume-burgers
burgers_mra.cpp:finite-volume-burgers-mra
burgers_os.cpp:finite-volume-burgers-os
)

include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)
# Create executables with PETSc
if(PETSC_FOUND)
    include_directories(${PETSC_INCLUDE_DIRS})
    find_package(MPI)

    foreach(demo_entry ${PETSC_DEMOS})
        string(REPLACE ":" ";" demo_parts ${demo_entry})
        list(GET demo_parts 0 source_file)
        list(GET demo_parts 1 executable_name)

        add_executable(${executable_name} ${source_file})
        target_link_libraries(${executable_name} samurai CLI11::CLI11 ${PETSC_LINK_LIBRARIES} ${MPI_LIBRARIES})
    endforeach()
endif()

# Create standard executables
foreach(demo_entry ${STANDARD_DEMOS})
    string(REPLACE ":" ";" demo_parts ${demo_entry})
    list(GET demo_parts 0 source_file)
    list(GET demo_parts 1 executable_name)

    add_executable(${executable_name} ${source_file})
    target_link_libraries(${executable_name} samurai CLI11::CLI11)
endforeach()

# Specific options for MSVC
if(MSVC)
    target_compile_options(finite-volume-level-set-from-scratch PUBLIC /bigobj)
    target_compile_options(finite-volume-level-set PUBLIC /bigobj)
endif()

# add_subdirectory(BZ)
