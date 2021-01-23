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

set(SAMURAI_BENCHMARKS
    benchmark_celllist_construction.cpp
    benchmark_search.cpp
    benchmark_set.cpp
    main.cpp
)

# foreach(filename IN LISTS SAMURAI_BENCHMARKS)
#     string(REPLACE ".cpp" "" targetname ${filename})
#     add_executable(${targetname} ${COMMON_BASE} ${filename} ${SAMURAI_HEADERS})
#     target_include_directories(${targetname} PRIVATE ${SAMURAI_INCLUDE_DIR})
#     target_link_libraries(${targetname} samurai benchmark::benchmark)
#     # gbenchk_add_tests(TARGET ${targetname})
# endforeach()

add_executable(bench_samurai ${SAMURAI_BENCHMARKS})
# target_include_directories(bench_samurai PRIVATE ${SAMURAI_INCLUDE_DIR})
target_link_libraries(bench_samurai samurai benchmark::benchmark)

# target_include_directories(bench_samurai_lib PRIVATE ${SAMURAI_INCLUDE_DIR})
# # if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
# #     add_dependencies(test_samurai_lib gtest_main)
# # endif()
# target_link_libraries(bench_samurai_lib samurai benchmark::benchmark)

# # set(SAMURAI_BENCHMARK_TARGET benchmark_samurai)
# # add_executable(${SAMURAI_BENCHMARK_TARGET} EXCLUDE_FROM_ALL ${SAMURAI_BENCHMARK} ${SAMURAI_HEADERS})
# add_executable(benchmark_set benchmark_set.cpp)
# target_link_libraries(benchmark_set samurai benchmark::benchmark)

# add_custom_target(xbenchmark
#     COMMAND benchmark_samurai
#     DEPENDS ${SAMURAI_BENCHMARK_TARGET})
