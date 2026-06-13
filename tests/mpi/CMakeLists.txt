# MPI tests: one GoogleTest executable per file, each registered with CTest
# for 2, 3 and 4 processes (3 catches asymmetries that 2 and 4 hide).
# Only configured when WITH_MPI=ON (see tests/CMakeLists.txt).

find_package(MPI REQUIRED COMPONENTS CXX)

set(MPI_TEST_PROCS 2 3 4)

function(add_mpi_gtest name)
    add_executable(${name} main.cpp ${name}.cpp)
    target_include_directories(${name} PRIVATE ${SAMURAI_INCLUDE_DIR})

    if(MSVC)
        target_compile_options(${name} PUBLIC /bigobj)
    endif()

    target_link_libraries(${name} samurai gtest)

    foreach(np IN LISTS MPI_TEST_PROCS)
        add_test(NAME ${name}_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                         $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(${name}_np${np} PROPERTIES PROCESSORS ${np} TIMEOUT 120)
    endforeach()
endfunction()

add_mpi_gtest(test_lb_migration)
add_mpi_gtest(test_lb_metrics)
add_mpi_gtest(test_lb_sfc)
add_mpi_gtest(test_lb_diffusion)
add_mpi_gtest(test_lb_ghosts)

# Graph construction test (always built with MPI; no external library needed)
add_mpi_gtest(test_lb_graph)

# ParMETIS tests (only if ParMETIS is available)
if(SAMURAI_WITH_PARMETIS)
    add_mpi_gtest(test_lb_metis)
    target_compile_definitions(test_lb_metis PRIVATE SAMURAI_WITH_PARMETIS)
    target_link_libraries(test_lb_metis ParMETIS::ParMETIS ParMETIS::Metis)

endif()

# PT-Scotch tests (only if PT-Scotch is available)
if(SAMURAI_WITH_PTSCOTCH)
    add_mpi_gtest(test_lb_scotch)
    target_compile_definitions(test_lb_scotch PRIVATE SAMURAI_WITH_PTSCOTCH)
    target_link_libraries(test_lb_scotch PTScotch::PTScotch PTScotch::Scotch)
endif()
