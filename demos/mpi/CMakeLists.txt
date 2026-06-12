# MPI-only demos (require WITH_MPI=ON).

if(NOT WITH_MPI)
    message(STATUS "demos/mpi skipped (WITH_MPI=OFF)")
    return()
endif()

add_executable(mpi-load-balancing-2d load_balancing_2d.cpp)
target_link_libraries(mpi-load-balancing-2d samurai)
