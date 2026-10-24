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
#
# samurai is an INTERFACE target: it compiles nothing, but its compile
# definitions (SAMURAI_ENABLE_INLINE, SAMURAI_FORCE_INLINE) decide how the set
# algebra is inlined. The binaries that use only the interval structures and
# the set algebra take it too, rather than its include directories, so that
# they measure the algebra as an application compiles it.
set(SAMURAI_BENCHMARK_SOURCES
    benchmark_celllist_construction.cpp
    benchmark_detail_computation.cpp
    benchmark_dynamic_set.cpp
    benchmark_fv_scheme.cpp
    benchmark_search.cpp
    benchmark_set_call_sites.cpp
    benchmark_set_graduation.cpp
    benchmark_set_library_ops.cpp
    benchmark_set_mesh_shape.cpp
    benchmark_set_operators.cpp
    benchmark_set_projection.cpp
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

# bench_set_call_sites, bench_dynamic_set and bench_set_graduation read the
# sets of the MR-adapted meshes from files, so they compile with the interval
# structures and the set algebra alone. generate_set_algebra_inputs writes the
# files. The build runs it when the directory has no stamp and when the
# sources that decide the content of the files change: the generator, the
# meshes (mr_fixtures.hpp), the list of meshes and the file format
# (site_inputs.hpp), the sites, which build the meshes of the ranks of a
# partition (set_call_sites.hpp), and those that decide the reference
# checksums of the sites (call_site_benchmarks.hpp, set_modes.hpp,
# site_checksum.hpp). It does not run it again when the library changes, so
# the inputs and the reference stay the same while the set algebra does. With
# SAMURAI_BENCHMARK_GENERATE_SET_ALGEBRA_INPUTS=OFF building these binaries no
# longer builds the generator, which is then run by hand.
# SAMURAI_SET_ALGEBRA_INPUTS in the environment overrides the directory at run
# time.
#
# SAMURAI_BENCHMARK_SET_ALGEBRA_SWEEP=ON adds the sweep over the level: the
# synthetic benchmarks run at a range of levels, and the generator writes and
# the call sites read one more mesh per max level (bench::sweep_cases). It has
# its own stamp, so that turning it on generates the meshes it adds.
option(SAMURAI_BENCHMARK_GENERATE_SET_ALGEBRA_INPUTS "Generate the set algebra inputs when building the benchmarks that read them" ON)
option(SAMURAI_BENCHMARK_SET_ALGEBRA_SWEEP "Run the set algebra benchmarks over a range of levels" OFF)
set(SAMURAI_SET_ALGEBRA_INPUTS_DIR ${CMAKE_CURRENT_BINARY_DIR}/set_algebra_inputs)
add_executable(generate_set_algebra_inputs generate_set_algebra_inputs.cpp)
target_link_libraries(generate_set_algebra_inputs samurai benchmark::benchmark)
set(SAMURAI_SET_ALGEBRA_INPUT_READERS bench_set_call_sites bench_dynamic_set bench_set_graduation)
foreach(target generate_set_algebra_inputs ${SAMURAI_SET_ALGEBRA_INPUT_READERS})
    target_compile_definitions(${target} PRIVATE SAMURAI_SET_ALGEBRA_INPUTS_DIR="${SAMURAI_SET_ALGEBRA_INPUTS_DIR}")
endforeach()
set(SAMURAI_SET_ALGEBRA_STAMP generated.stamp)
if(SAMURAI_BENCHMARK_SET_ALGEBRA_SWEEP)
    set(SAMURAI_SET_ALGEBRA_STAMP generated-sweep.stamp)
    foreach(target generate_set_algebra_inputs ${SAMURAI_SET_ALGEBRA_INPUT_READERS}
                   bench_set_operators bench_set_projection bench_set_mesh_shape bench_set_library_ops)
        target_compile_definitions(${target} PRIVATE SAMURAI_SET_ALGEBRA_SWEEP)
    endforeach()
endif()
if(SAMURAI_BENCHMARK_GENERATE_SET_ALGEBRA_INPUTS)
    # The generator is named in COMMAND, which builds it first, and not in
    # DEPENDS, which would also run it again each time it is rebuilt, that is
    # each time a library header changes. DEPENDS lists its sources instead.
    add_custom_command(
        OUTPUT ${SAMURAI_SET_ALGEBRA_INPUTS_DIR}/${SAMURAI_SET_ALGEBRA_STAMP}
        COMMAND generate_set_algebra_inputs
        COMMAND ${CMAKE_COMMAND} -E touch ${SAMURAI_SET_ALGEBRA_INPUTS_DIR}/${SAMURAI_SET_ALGEBRA_STAMP}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/call_site_benchmarks.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_set_algebra_inputs.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mr_fixtures.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/set_algebra_backend.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/set_algebra_fixtures.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/set_call_sites.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/set_modes.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/site_checksum.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/site_inputs.hpp
        COMMENT "Generating the set algebra inputs in ${SAMURAI_SET_ALGEBRA_INPUTS_DIR}"
    )
    add_custom_target(set_algebra_inputs DEPENDS ${SAMURAI_SET_ALGEBRA_INPUTS_DIR}/${SAMURAI_SET_ALGEBRA_STAMP})
    foreach(target IN LISTS SAMURAI_SET_ALGEBRA_INPUT_READERS)
        add_dependencies(${target} set_algebra_inputs)
    endforeach()
endif()

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
