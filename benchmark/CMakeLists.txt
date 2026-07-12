cmake_minimum_required(VERSION 3.11)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(samurai-benchmark)

    find_package(samurai REQUIRED CONFIG)
    set(SAMURAI_INCLUDE_DIR ${samurai_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing benchmark build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

include(addGoogleTest)
include(addGoogleBench)

find_package(Threads)

# Self-sufficient benchmarks: one source file, one executable, each file
# providing its own main (BENCHMARK_MAIN() or, where samurai::initialize()/
# finalize() is required, a hand-written one), linking only samurai and
# Google Benchmark. To add a benchmark, add its source file to this list;
# samurai_add_benchmark() below turns benchmark_<name>.cpp into the
# bench_<name> executable.
set(SAMURAI_BENCHMARK_SOURCES
    benchmark_celllist_construction.cpp
    benchmark_detail_computation.cpp
    benchmark_fv_scheme.cpp
    benchmark_search.cpp
    benchmark_set.cpp
    benchmark_update_fields.cpp
    benchmark_update_ghost.cpp
)

function(samurai_add_benchmark source)
    get_filename_component(name ${source} NAME_WE)
    string(REPLACE "benchmark_" "bench_" target ${name})
    add_executable(${target} ${source})
    target_link_libraries(${target} samurai benchmark::benchmark)
endfunction()

foreach(source IN LISTS SAMURAI_BENCHMARK_SOURCES)
    samurai_add_benchmark(${source})
endforeach()

# Load balancing micro-benchmarks: MPI-only, its own MPI-aware main(), and
# optional extra links (ParMETIS / PT-Scotch), so it is built outside the
# generic loop above.
if(WITH_MPI)
    add_executable(bench_load_balancing benchmark_load_balancing.cpp)
    target_include_directories(bench_load_balancing PRIVATE ${SAMURAI_INCLUDE_DIR})
    target_link_libraries(bench_load_balancing samurai benchmark::benchmark Threads::Threads)

    if(SAMURAI_WITH_PARMETIS)
        target_compile_definitions(bench_load_balancing PRIVATE SAMURAI_WITH_PARMETIS)
        target_link_libraries(bench_load_balancing ParMETIS::ParMETIS ParMETIS::Metis)
    endif()

    if(SAMURAI_WITH_PTSCOTCH)
        target_compile_definitions(bench_load_balancing PRIVATE SAMURAI_WITH_PTSCOTCH)
        target_link_libraries(bench_load_balancing PTScotch::PTScotch PTScotch::Scotch)
    endif()
endif()
