cmake_minimum_required(VERSION 3.11)

project(samurai)

set(SAMURAI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Versionning
# ===========

file(STRINGS "${SAMURAI_INCLUDE_DIRS}/samurai/samurai_config.hpp" samurai_version_defines
     REGEX "#define SAMURAI_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${samurai_version_defines})
    if(ver MATCHES "#define SAMURAI_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(SAMURAI_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${SAMURAI_VERSION_MAJOR}.${SAMURAI_VERSION_MINOR}.${SAMURAI_VERSION_PATCH})
message(STATUS "Building samurai v${${PROJECT_NAME}_VERSION}")

# Dependencies
# ============

find_package(xtensor REQUIRED)
find_package(HDF5 REQUIRED)
find_package(xtensor-io REQUIRED)
find_package(pugixml REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt)

# Target
# ======

add_library(samurai INTERFACE)
target_include_directories(
    samurai
    INTERFACE $<BUILD_INTERFACE:${SAMURAI_INCLUDE_DIRS}>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(samurai INTERFACE xtensor pugixml spdlog hdf5 fmt::fmt)

OPTION(BUILD_BENCHMARKS "samurai benchmark suite" OFF)
OPTION(BUILD_TESTS "samurai test suite" OFF)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

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

install(DIRECTORY ${SAMURAI_INCLUDE_DIRS}/samurai
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(SAMURAI_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for samuraiConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${SAMURAI_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
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
