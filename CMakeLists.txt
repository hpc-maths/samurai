cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

# Add project_options v0.25.2
# https://github.com/aminya/project_options
include(FetchContent)
FetchContent_Declare(_project_options URL https://github.com/aminya/project_options/archive/refs/tags/v0.25.2.zip)
FetchContent_MakeAvailable(_project_options)
include(${_project_options_SOURCE_DIR}/Index.cmake)

# install vcpkg dependencies: - should be called before defining project()
option(ENABLE_VCPKG "Use vcpkg to install dependencies" OFF)
option(ENABLE_CONAN_OPTION "Use Conan to install dependencies" OFF)
if (${ENABLE_VCPKG})
  run_vcpkg()
endif()

if (${ENABLE_CONAN_OPTION})
  set(ENABLE_CONAN "ENABLE_CONAN")
endif()

# Enable sanitizers and static analyzers when running the tests
option(CLANG_TIDY "Activate clang-tidy" OFF)
option(CPPCHECK "Activate cppcheck" OFF)
option(IWYU "Activate include-what-you-use" OFF)
option(SANITIZERS "Activate sanitizers" OFF)
option(ENABLE_COVERAGE "Activate coverage" OFF)

project(samurai VERSION 0.1.0 LANGUAGES CXX C)

SET(FEATURES)
if (${CLANG_TIDY})
  LIST(APPEND FEATURES ENABLE_CLANG_TIDY)
endif()

if (${CPPCHECK})
  LIST(APPEND FEATURES ENABLE_CPPCHECK)
endif()

if (${IWYU})
  LIST(APPEND FEATURES ENABLE_INCLUDE_WHAT_YOU_USE)
endif()

if (${SANITIZERS})
  LIST(APPEND FEATURES ENABLE_SANITIZER_ADDRESS)
  LIST(APPEND FEATURES ENABLE_SANITIZER_UNDEFINED_BEHAVIOR)
endif()

if (${COVERAGE})
  LIST(APPEND FEATURES ENABLE_COVERAGE)
endif()

message(STATUS "Available FEATURES: ${FEATURES}")
# set(coucous ENABLE_CLANG_TIDY ENABLE_CPPCHECK)
# Initialize project_options variable related to this project
# This overwrites `project_options` and sets `project_warnings`
# uncomment to enable the options. Some of them accept one or more inputs:
project_options(
  # ENABLE_CACHE
  # ${ENABLE_CPPCHECK}
  # ${ENABLE_CLANG_TIDY}
  ${FEATURES}
  ENABLE_VS_ANALYSIS
  ${ENABLE_CONAN}
  # ENABLE_INTERPROCEDURAL_OPTIMIZATION
  # ENABLE_NATIVE_OPTIMIZATION
  #   ${ENABLE_DOXYGEN}
  # ${ENABLE_COVERAGE}
  # ${ENABLE_SANITIZER_ADDRESS}
  # ${ENABLE_SANITIZER_UNDEFINED_BEHAVIOR}
  # ENABLE_SANITIZER_THREAD
  # ENABLE_SANITIZER_MEMORY
  # ENABLE_PCH
  # PCH_HEADERS
    # WARNINGS_AS_ERRORS
  # ${ENABLE_INCLUDE_WHAT_YOU_USE}
  # ENABLE_USER_LINKER
  #   ENABLE_BUILD_WITH_TIME_TRACE
  # ENABLE_UNITY
  # CONAN_PROFILE ${profile_path}  # passes a profile to conan: see https://docs.conan.io/en/latest/reference/profiles.html
)

add_library(samurai INTERFACE)
target_link_libraries(samurai INTERFACE project_options project_warnings)

# Includes
set(INCLUDE_DIR "include") # must be relative paths
target_include_directories(samurai INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}>"
                                             "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Find dependencies:
set(DEPENDENCIES_CONFIGURED xtensor HighFive pugixml fmt)

if (${ENABLE_VCPKG})
  list(APPEND DEPENDENCIES_CONFIGURED hdf5)
endif()

# Force HIGHFIVE_USE_INSTALL_DEPS option to be true
option(HIGHFIVE_USE_INSTALL_DEPS "End applications by default use detected dependencies here" ON)

foreach(DEPENDENCY ${DEPENDENCIES_CONFIGURED})
  find_package(${DEPENDENCY} CONFIG REQUIRED)
endforeach()


# Link dependencies:
target_link_system_libraries(
  samurai
  INTERFACE
  xtensor
  HighFive
  pugixml::pugixml
  fmt::fmt
)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

OPTION(BUILD_BENCHMARKS "samurai benchmark suite" OFF)
OPTION(BUILD_DEMOS "samurai build all demos" OFF)
OPTION(BUILD_TESTS "samurai test suite" OFF)
OPTION(WITH_STATS "samurai mesh stats" OFF)

if (WITH_STATS)
  find_package(nlohmann_json REQUIRED)
  target_link_libraries(samurai INTERFACE nlohmann_json::nlohmann_json)
  target_compile_definitions(samurai INTERFACE WITH_STATS)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

if(BUILD_TESTS)
  add_subdirectory(test)
endif()

if(BUILD_DEMOS)
    add_subdirectory(demos)
else()
    add_subdirectory(demos EXCLUDE_FROM_ALL)
endif()

# Package the project
package_project(
  TARGETS samurai project_options project_warnings
  INTERFACE_DEPENDENCIES_CONFIGURED ${DEPENDENCIES_CONFIGURED}
  INTERFACE_INCLUDES ${INCLUDE_DIR}
)
