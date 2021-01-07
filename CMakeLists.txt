cmake_minimum_required(VERSION 3.11)

project(samurai)

set(SAMURAI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Versionning
# ===========

file(STRINGS "${SAMURAI_INCLUDE_DIR}/samurai/samurai_config.hpp" samurai_version_defines
     REGEX "#define SAMURAI_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${samurai_version_defines})
    if(ver MATCHES "#define SAMURAI_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(SAMURAI_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${SAMURAI_VERSION_MAJOR}.${SAMURAI_VERSION_MINOR}.${SAMURAI_VERSION_PATCH})
message(STATUS "Building samurai v${${PROJECT_NAME}_VERSION}")


find_package(xtensor REQUIRED)
find_package(HDF5 REQUIRED)
find_package(xtensor-io REQUIRED)
find_package(pugixml REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt)

set(SAMURAI_HEADERS
    ${SAMURAI_INCLUDE_DIR}/samurai/algorithm.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/bc.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/box.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/cell.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/cell_array.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/cell_list.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/field.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/field_expression.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/hdf5.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/interval.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/level_cell_array.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/level_cell_list.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/list_of_intervals.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mesh.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/adapt.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/cell_flag.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/coarsening.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/criteria.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/harten.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/mesh.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/operators.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/pred_and_proj.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/prediction.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/mr/refinement.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/operators_base.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/samurai_config.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/static_algorithm.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/stencil_field.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/subset/node_op.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/subset/subset_node.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/subset/subset_op.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/subset/subset_op_base.hpp
    ${SAMURAI_INCLUDE_DIR}/samurai/utils.hpp
)

add_library(samurai INTERFACE)
target_include_directories(samurai INTERFACE $<BUILD_INTERFACE:${SAMURAI_INCLUDE_DIR}>
                                             $<INSTALL_INTERFACE:include>)

target_link_libraries(samurai INTERFACE xtensor pugixml spdlog hdf5 fmt::fmt)

OPTION(BUILD_TESTS "samurai test suite" OFF)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

add_subdirectory(demos)

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS samurai
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${SAMURAI_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/samurai)

set(SAMURAI_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for samuraiConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${SAMURAI_CMAKECONFIG_INSTALL_DIR})

# xtensor is header-only and does not depend on the architecture.
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
# Remove CMAKE_SIZEOF_VOID_P from xtensorConfigVersion.cmake so that an samuraiConfig.cmake
set(_SAMURAI_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_SAMURAI_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${SAMURAI_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${SAMURAI_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
