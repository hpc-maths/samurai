include(addGoogleTest)
include_directories(${GTEST_INCLUDE_DIRS} SYSTEM)

find_package(rapidcheck QUIET)
message(STATUS "Found Rapidcheck: ${rapidcheck_FOUND}")

find_package(Threads)

# set(CMAKE_CXX_STANDARD 17)

# if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
#     #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion")
#     CHECK_CXX_COMPILER_FLAG("-std=c++17" HAS_CPP17_FLAG)

#     set(CMAKE_CXX_STANDARD_REQUIRED ON)
#     set(CMAKE_CXX_EXTENSIONS OFF)
# endif()

# if(MSVC)
#     add_definitions(-D_CRT_SECURE_NO_WARNINGS)
#     add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /wd4800")
#     set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
# endif()

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
    test_portion.cpp
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
