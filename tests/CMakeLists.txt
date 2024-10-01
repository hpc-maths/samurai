include(addGoogleTest)
include_directories(${GTEST_INCLUDE_DIRS} SYSTEM)

find_package(rapidcheck QUIET)
message(STATUS "Found Rapidcheck: ${rapidcheck_FOUND}")

find_package(Threads)

set(COMMON_BASE
    main.cpp
)

set(SAMURAI_TESTS
    test_adapt.cpp
    test_bc.cpp
    test_box.cpp
    test_cell.cpp
    test_cell_array.cpp
    test_cell_list.cpp
    test_field.cpp
    test_for_each.cpp
    test_graduation.cpp
    test_interval.cpp
    test_level_cell_list.cpp
    test_list_of_intervals.cpp
    test_periodic.cpp
    test_portion.cpp
    test_utils.cpp
    test_sfc.cpp
    test_mrmesh.cpp
    test_loadbalancing.cpp
)

if(rapidcheck_FOUND)
    list(APPEND SAMURAI_TESTS test_operator_set.cpp)
endif()

foreach(filename IN LISTS SAMURAI_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} ${COMMON_BASE} ${filename} ${SAMURAI_HEADERS})
    target_include_directories(${targetname} PRIVATE ${SAMURAI_INCLUDE_DIR})

    if(MSVC)
        target_compile_options(${targetname} PUBLIC /bigobj)
    endif()

    if(rapidcheck_FOUND)
        target_link_libraries(${targetname} samurai gtest_main gtest rapidcheck rapidcheck_gtest)
    else()
        target_link_libraries(${targetname} samurai gtest_main gtest)
    endif()
endforeach()

add_executable(test_samurai_lib ${COMMON_BASE} ${SAMURAI_TESTS} ${SAMURAI_HEADERS})
target_include_directories(test_samurai_lib PRIVATE ${SAMURAI_INCLUDE_DIR})

if(MSVC)
    target_compile_options(test_samurai_lib PUBLIC /bigobj)
endif()

if(rapidcheck_FOUND)
    target_link_libraries(test_samurai_lib samurai gtest_main gtest rapidcheck rapidcheck_gtest)
else()
    target_link_libraries(test_samurai_lib samurai gtest_main gtest)
endif()
