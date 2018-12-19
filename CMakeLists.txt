cmake_minimum_required(VERSION 3.0)

project(mure)

set(MURE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(xtensor REQUIRED)
find_package(xtensor-io REQUIRED)

set(MURE_HEADERS
    ${MURE_INCLUDE_DIR}/mure/box.hpp
    ${MURE_INCLUDE_DIR}/mure/cell.hpp
    ${MURE_INCLUDE_DIR}/mure/cell_array.hpp
    ${MURE_INCLUDE_DIR}/mure/cell_list.hpp
    ${MURE_INCLUDE_DIR}/mure/field.hpp
    ${MURE_INCLUDE_DIR}/mure/func_node.hpp
    ${MURE_INCLUDE_DIR}/mure/hdf5.hpp
    ${MURE_INCLUDE_DIR}/mure/interval.hpp
    ${MURE_INCLUDE_DIR}/mure/intervals_operator.hpp
    ${MURE_INCLUDE_DIR}/mure/level_cell_array.hpp
    ${MURE_INCLUDE_DIR}/mure/level_cell_list.hpp
    ${MURE_INCLUDE_DIR}/mure/mesh.hpp
    ${MURE_INCLUDE_DIR}/mure/mr_config.hpp
    ${MURE_INCLUDE_DIR}/mure/static_algorithm.hpp
    ${MURE_INCLUDE_DIR}/mure/subset.hpp
    ${MURE_INCLUDE_DIR}/mure/tuple.hpp
)

add_library(mure INTERFACE)
target_include_directories(mure INTERFACE $<BUILD_INTERFACE:${MURE_INCLUDE_DIR}>
                                          $<INSTALL_INTERFACE:include>)
target_link_libraries(mure INTERFACE xtensor hdf5)

OPTION(BUILD_TESTS "mure test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

# if(BUILD_TESTS)
#     add_subdirectory(test)
# endif()

add_subdirectory(demos)
