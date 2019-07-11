cmake_minimum_required(VERSION 3.0)

project(mure)

set(MURE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS "${MURE_INCLUDE_DIR}/mure/mure_config.hpp" mure_version_defines
     REGEX "#define MURE_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${mure_version_defines})
    if(ver MATCHES "#define MURE_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(MURE_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${MURE_VERSION_MAJOR}.${MURE_VERSION_MINOR}.${MURE_VERSION_PATCH})
message(STATUS "Building mure v${${PROJECT_NAME}_VERSION}")


find_package(xtensor REQUIRED)
find_package(xtensor-io REQUIRED)
find_package(pugixml REQUIRED)

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
target_link_libraries(mure INTERFACE xtensor pugixml hdf5)
# target_link_libraries(mure INTERFACE xtensor)

OPTION(BUILD_TESTS "mure test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)

if(DOWNLOAD_GTEST)
    set(BUILD_TESTS ON)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

add_subdirectory(demos)

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS mure
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${MURE_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mure)

set(MURE_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for mureConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${MURE_CMAKECONFIG_INSTALL_DIR})

# xtensor is header-only and does not depend on the architecture.
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
# Remove CMAKE_SIZEOF_VOID_P from xtensorConfigVersion.cmake so that an mureConfig.cmake
set(_MURE_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_MURE_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${MURE_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${MURE_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
