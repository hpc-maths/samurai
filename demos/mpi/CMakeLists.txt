# MPI-only demos (require WITH_MPI=ON).

if(NOT WITH_MPI)
    message(STATUS "demos/mpi skipped (WITH_MPI=OFF)")
    return()
endif()

# The same source builds the 2D and 3D comparative load balancing demos; the
# dimension is fixed at compile time via SAMURAI_LB_DIM.
foreach(d 2 3)
    set(target mpi-load-balancing-${d}d)
    add_executable(${target} load_balancing.cpp)
    target_compile_definitions(${target} PRIVATE SAMURAI_LB_DIM=${d})
    target_link_libraries(${target} samurai)

    if(SAMURAI_WITH_PARMETIS)
        target_compile_definitions(${target} PRIVATE SAMURAI_WITH_PARMETIS)
        target_link_libraries(${target} ParMETIS::ParMETIS ParMETIS::Metis)
    endif()

    if(SAMURAI_WITH_PTSCOTCH)
        target_compile_definitions(${target} PRIVATE SAMURAI_WITH_PTSCOTCH)
        target_link_libraries(${target} PTScotch::PTScotch PTScotch::Scotch)
    endif()
endforeach()

# Visual validation of the parallel ghost-update robustness suite: rebuilds
# every mesh + decomposition swept by tests/mpi/test_ghost_update_parallel.cpp
# and dumps them to HDF5/XDMF. The case catalog (ghost_cases.hpp) is shared with
# the test, so this tool visualises exactly the cases the test validates.
set(ghost_cases_target mpi-ghost-cases)
add_executable(${ghost_cases_target} ghost_cases/dump_ghost_cases.cpp)
target_include_directories(${ghost_cases_target} PRIVATE ${CMAKE_SOURCE_DIR}/tests/mpi)
target_link_libraries(${ghost_cases_target} samurai)

# Interactive tube demo: play with the number of circles and the load balancing
# strategy on a non-square (1 x length) domain. 2D only.
set(tube_target mpi-load-balancing-tube)
add_executable(${tube_target} load_balancing_tube.cpp)
target_link_libraries(${tube_target} samurai)

if(SAMURAI_WITH_PARMETIS)
    target_compile_definitions(${tube_target} PRIVATE SAMURAI_WITH_PARMETIS)
    target_link_libraries(${tube_target} ParMETIS::ParMETIS ParMETIS::Metis)
endif()

if(SAMURAI_WITH_PTSCOTCH)
    target_compile_definitions(${tube_target} PRIVATE SAMURAI_WITH_PTSCOTCH)
    target_link_libraries(${tube_target} PTScotch::PTScotch PTScotch::Scotch)
endif()
