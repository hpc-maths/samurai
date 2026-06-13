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
