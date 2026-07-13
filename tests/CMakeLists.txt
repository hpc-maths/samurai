include(addGoogleTest)
include_directories(${GTEST_INCLUDE_DIRS} SYSTEM)

find_package(rapidcheck QUIET)
message(STATUS "Found Rapidcheck: ${rapidcheck_FOUND}")

find_package(Threads)

# PETSc support for the tests (mirrors demos/FiniteVolume/CMakeLists.txt). Enables
# the explicit/implicit consistency tests guarded by SAMURAI_WITH_PETSC.
if(WITH_PETSC)
    message(STATUS "Building tests with PETSc support")
    include(CMakeFindDependencyMacro)
    find_dependency(PkgConfig)
    pkg_check_modules(PETSC REQUIRED PETSc)
    include_directories(${PETSC_INCLUDE_DIRS})
endif()

# Adds the PETSc compile definition and libraries to a test target when enabled.
function(samurai_test_petsc target)
    if(WITH_PETSC)
        target_compile_definitions(${target} PUBLIC SAMURAI_WITH_PETSC)
        target_link_libraries(${target} ${PETSC_LINK_LIBRARIES})
    endif()
endfunction()

set(COMMON_BASE
    main.cpp
)

set(SAMURAI_TESTS
    test_adapt.cpp
    test_bc.cpp
    test_bc_ghost_values.cpp
    test_box.cpp
    test_cell.cpp
    test_cell_array.cpp
    test_cell_list.cpp
    test_corner_projection.cpp
    test_domain_with_hole.cpp
    test_field.cpp
    test_find.cpp
    test_for_each.cpp
    test_fv_operators.cpp
    test_ghost_update.cpp
    test_graduation.cpp
    test_hdf5.cpp
    test_interval.cpp
    test_level_cell_list.cpp
    test_list_of_intervals.cpp
    test_load_balancing_weight.cpp
    test_mra.cpp
    test_periodic.cpp
    test_portion.cpp
    test_projection_prediction_roundtrip.cpp
    test_restart.cpp
    test_scaling.cpp
    test_sfc_curves.cpp
    test_static_dispatch.cpp
    test_stencil.cpp
    test_subdomain_bbox.cpp
    test_subset.cpp
    test_timers.cpp
    test_utils.cpp
    test_version.cpp
)

if(rapidcheck_FOUND)
    list(APPEND SAMURAI_TESTS test_operator_set.cpp)
endif()

if (SPLIT_TESTS)
    foreach(filename IN LISTS SAMURAI_TESTS)
        string(REPLACE ".cpp" "" targetname ${filename})
        add_executable(${targetname} ${COMMON_BASE} ${filename} ${SAMURAI_HEADERS})
        target_include_directories(${targetname} PRIVATE ${SAMURAI_INCLUDE_DIR})

        if(MSVC)
            target_compile_options(${targetname} PUBLIC /bigobj)
        endif()

        if(rapidcheck_FOUND)
            target_link_libraries(${targetname} samurai gtest_main gtest rapidcheck rapidcheck_gtest)
        else()
            target_link_libraries(${targetname} samurai gtest_main gtest)
        endif()
    endforeach()
endif()

add_executable(test_samurai_lib ${COMMON_BASE} ${SAMURAI_TESTS} ${SAMURAI_HEADERS})
target_include_directories(test_samurai_lib PRIVATE ${SAMURAI_INCLUDE_DIR})

if(MSVC)
    target_compile_options(test_samurai_lib PUBLIC /bigobj)
endif()

if(rapidcheck_FOUND)
    target_link_libraries(test_samurai_lib samurai gtest_main gtest rapidcheck rapidcheck_gtest)
else()
    target_link_libraries(test_samurai_lib samurai gtest_main gtest)
endif()

# The PETSc-backed tests (SAMURAI_WITH_PETSC) live in a dedicated executable.
# They cannot share the process with the rest of the suite: PetscFinalize() (called
# by samurai::finalize() in some tests) tears down MPI, which cannot be restarted,
# so a later PETSc test in the same binary would abort. Isolating them also keeps
# SAMURAI_WITH_PETSC off the main suite, avoiding pulling PETSc into every TU.
if(WITH_PETSC)
    add_executable(test_fv_operators_petsc ${COMMON_BASE} test_fv_operators.cpp ${SAMURAI_HEADERS})
    target_include_directories(test_fv_operators_petsc PRIVATE ${SAMURAI_INCLUDE_DIR})
    if(MSVC)
        target_compile_options(test_fv_operators_petsc PUBLIC /bigobj)
    endif()
    target_link_libraries(test_fv_operators_petsc samurai gtest_main gtest)
    samurai_test_petsc(test_fv_operators_petsc)
endif()

# Standalone tools (no gtest): regenerate the small samurai files consumed by
# test_paraview_reader.py / test_yt_reader.py. Not tracked in git (they are
# gitignored .h5 files), so CI must build and run these before running those tests.
add_executable(generate_paraview_reference ${CMAKE_CURRENT_SOURCE_DIR}/../tools/paraview/tests/generate_reference.cpp)
target_include_directories(generate_paraview_reference PRIVATE ${SAMURAI_INCLUDE_DIR})
target_link_libraries(generate_paraview_reference samurai)

add_executable(generate_yt_reference ${CMAKE_CURRENT_SOURCE_DIR}/../tools/yt/tests/generate_reference.cpp)
target_include_directories(generate_yt_reference PRIVATE ${SAMURAI_INCLUDE_DIR})
target_link_libraries(generate_yt_reference samurai)

if(WITH_MPI)
    add_subdirectory(mpi)
endif()
