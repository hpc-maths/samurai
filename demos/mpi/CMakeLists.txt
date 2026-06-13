# MPI-only demos (require WITH_MPI=ON).

if(NOT WITH_MPI)
    message(STATUS "demos/mpi skipped (WITH_MPI=OFF)")
    return()
endif()

add_executable(mpi-load-balancing-2d load_balancing_2d.cpp)
target_link_libraries(mpi-load-balancing-2d samurai)

if(SAMURAI_WITH_PARMETIS)
    target_compile_definitions(mpi-load-balancing-2d PRIVATE SAMURAI_WITH_PARMETIS)
    target_link_libraries(mpi-load-balancing-2d ParMETIS::ParMETIS ParMETIS::Metis)
endif()

if(SAMURAI_WITH_PTSCOTCH)
    target_compile_definitions(mpi-load-balancing-2d PRIVATE SAMURAI_WITH_PTSCOTCH)
    target_link_libraries(mpi-load-balancing-2d PTScotch::PTScotch PTScotch::Scotch)
endif()
