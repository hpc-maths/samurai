if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(samurai-test)

    find_package(samurai REQUIRED CONFIG)
    set(SAMURAI_INCLUDE_DIR ${SAMURAI_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

find_package(g QUIET)

include(addGoogleTest)
# include(addRapidCheck)

find_package(rapidcheck QUIET)
find_package(Threads)

include_directories(${GTEST_INCLUDE_DIRS} SYSTEM)
message(STATUS ${rapidcheck_FOUND})

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
        message(FATAL_ERROR "Unsupported compiler -- samurai requires C++14 support!")
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

set(COMMON_BASE
    main.cpp
)

set(SAMURAI_TESTS
    test_box.cpp
    test_cell.cpp
    test_cell_list.cpp
    test_interval.cpp
    test_level_cell_list.cpp
    test_list_of_intervals.cpp
    # test_param.cpp
    # test_operator_set.cpp
)

if (rapidcheck_FOUND)
    list(APPEND SAMURAI_TESTS test_operator_set.cpp)
endif()

foreach(filename IN LISTS SAMURAI_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} ${COMMON_BASE} ${filename} ${SAMURAI_HEADERS})
    target_include_directories(${targetname} PRIVATE ${SAMURAI_INCLUDE_DIR})
    if (rapidcheck_FOUND)
        target_link_libraries(${targetname} samurai gtest_main gtest rapidcheck rapidcheck_gtest)
    else()
        target_link_libraries(${targetname} samurai gtest_main gtest)
    endif()
endforeach()

add_executable(test_samurai_lib ${COMMON_BASE} ${SAMURAI_TESTS} ${SAMURAI_HEADERS})
target_include_directories(test_samurai_lib PRIVATE ${SAMURAI_INCLUDE_DIR})
if (rapidcheck_FOUND)
    target_link_libraries(test_samurai_lib samurai gtest_main gtest rapidcheck rapidcheck_gtest)
else()
    target_link_libraries(test_samurai_lib samurai gtest_main gtest)
endif()
