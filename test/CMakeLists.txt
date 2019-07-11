cmake_minimum_required(VERSION 3.1)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(mure-test)

    find_package(mure REQUIRED CONFIG)
    set(MURE_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
    CHECK_CXX_COMPILER_FLAG("-std=c++17" HAS_CPP17_FLAG)

    if (CPP17 AND HAS_CPP17_FLAG)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "Building with -std=c++17")
    elseif (HAS_CPP14_FLAG)
        set(CMAKE_CXX_STANDARD 14)
        message(STATUS "Building with -std=c++14")
    else()
        message(FATAL_ERROR "Unsupported compiler -- xtensor requires C++14 support!")
    endif()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /wd4800")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    if(DOWNLOAD_GTEST)
        # Download and unpack googletest at configure time
        configure_file(downloadGTest.cmake.in googletest-download/CMakeLists.txt)
    else()
        # Copy local source of googletest at configure time
        configure_file(copyGTest.cmake.in googletest-download/CMakeLists.txt)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL)

    set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
    set(GTEST_BOTH_LIBRARIES gtest_main gtest)
else()
    find_package(GTest REQUIRED)
endif()

find_package(Threads)

include_directories(${GTEST_INCLUDE_DIRS} SYSTEM)

set(COMMON_BASE
    main.cpp
)

set(MURE_TESTS
    test_box.cpp
    test_cell.cpp
    test_interval.cpp
    test_level_cell_list.cpp
    test_list_of_intervals.cpp
)

include(GoogleTest)

foreach(filename IN LISTS MURE_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} ${COMMON_BASE} ${filename} ${MURE_HEADERS})
    target_include_directories(${targetname} PRIVATE ${MURE_INCLUDE_DIR})
    target_link_libraries(${targetname} mure ${GTEST_BOTH_LIBRARIES})
    # target_link_libraries(${targetname} mure ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    # if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    #     add_dependencies(${targetname} gtest_main)
    # endif()
    gtest_add_tests(TARGET ${targetname})
    # add_custom_target(
    #     x${targetname}
    #     COMMAND ${targetname}
    #     DEPENDS ${targetname} ${filename} ${XMURE_HEADERS})
endforeach()

add_executable(test_mure_lib ${COMMON_BASE} ${MURE_TESTS} ${MURE_HEADERS})
target_include_directories(test_mure_lib PRIVATE ${MURE_INCLUDE_DIR})
if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    add_dependencies(test_mure_lib gtest_main)
endif()
target_link_libraries(test_mure_lib mure ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

# # add_custom_target(mure_test COMMAND test_mure_lib DEPENDS test_mure_lib)
# add_test(NAME mure_test COMMAND test_mure_lib DEPENDS test_mure_lib)