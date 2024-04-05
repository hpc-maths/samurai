cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# Set version
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" SAMURAI_VERSION)
  string(STRIP "${SAMURAI_VERSION}" SAMURAI_VERSION)
else()
  message(FATAL_ERROR "File ${CMAKE_CURRENT_SOURCE_DIR}/version.txt not found")
endif()

# Add project_options v0.33.0
# https://github.com/aminya/project_options
include(FetchContent)
FetchContent_Declare(_project_options URL https://github.com/aminya/project_options/archive/refs/tags/v0.33.0.zip)
FetchContent_MakeAvailable(_project_options)
include(${_project_options_SOURCE_DIR}/Index.cmake)

# install vcpkg dependencies: - should be called before defining project()
option(ENABLE_VCPKG "Use vcpkg to install dependencies" OFF)
option(ENABLE_CONAN_OPTION "Use Conan to install dependencies" OFF)

if(${ENABLE_VCPKG})
  run_vcpkg()
  set(VCPKG_BUILD_TYPE release)
endif()

if(${ENABLE_CONAN_OPTION})
  set(ENABLE_CONAN "ENABLE_CONAN")
endif()

# Enable sanitizers and static analyzers when running the tests
option(CLANG_TIDY "Activate clang-tidy" OFF)
option(CPPCHECK "Activate cppcheck" OFF)
option(IWYU "Activate include-what-you-use" OFF)
option(SANITIZERS "Activate sanitizers" OFF)
option(ENABLE_COVERAGE "Activate coverage" OFF)
option(WITH_MPI "Enable MPI" OFF)
option(WITH_OPENMP "Enable OpenMP" OFF)

set(FLUX_CONTAINER_LIST array xtensor)
set(FLUX_CONTAINER "xtensor" CACHE STRING "Container to store fluxes: ${FLUX_CONTAINER_LIST}")
set_property(CACHE FLUX_CONTAINER PROPERTY STRINGS ${FLUX_CONTAINER_LIST})

project(samurai VERSION ${SAMURAI_VERSION} LANGUAGES CXX C)

SET(FEATURES)

if(${CLANG_TIDY})
  LIST(APPEND FEATURES ENABLE_CLANG_TIDY)
endif()

if(${CPPCHECK})
  LIST(APPEND FEATURES ENABLE_CPPCHECK)
endif()

if(${IWYU})
  LIST(APPEND FEATURES ENABLE_INCLUDE_WHAT_YOU_USE)
endif()

if(${SANITIZERS})
  LIST(APPEND FEATURES ENABLE_SANITIZER_ADDRESS)
  LIST(APPEND FEATURES ENABLE_SANITIZER_UNDEFINED_BEHAVIOR)
endif()

if(${COVERAGE})
  LIST(APPEND FEATURES ENABLE_COVERAGE)
endif()

message(STATUS "Available FEATURES: ${FEATURES}")
project_options(
  ${FEATURES}
  ENABLE_VS_ANALYSIS
  ${ENABLE_CONAN}
)

add_library(samurai INTERFACE)
target_link_libraries(samurai INTERFACE project_options project_warnings)
set_target_properties(samurai
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

# Includes
set(INCLUDE_DIR "include") # must be relative paths
target_include_directories(samurai INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Find dependencies:
set(DEPENDENCIES_CONFIGURED xtensor HighFive pugixml fmt)

if(${ENABLE_VCPKG})
  list(APPEND DEPENDENCIES_CONFIGURED hdf5)
endif()

# Force HIGHFIVE_USE_INSTALL_DEPS option to be true
option(HIGHFIVE_USE_INSTALL_DEPS "End applications by default use detected dependencies here" OFF)

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

if(${WITH_OPENMP})
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(samurai INTERFACE OpenMP::OpenMP_CXX)
      target_compile_definitions(samurai INTERFACE SAMURAI_WITH_OPENMP)
  else()
      message(FATAL_ERROR "OpenMP not found")
  endif()
endif()

if(${WITH_MPI})
  find_package(Boost REQUIRED COMPONENTS serialization mpi)
  target_link_system_libraries(
    samurai
    INTERFACE
    Boost::serialization
    Boost::mpi
  )
  target_compile_definitions(samurai INTERFACE SAMURAI_WITH_MPI)
endif()


target_compile_features(samurai INTERFACE cxx_std_17)

# target_compile_definitions(samurai INTERFACE _HAS_AUTO_PTR_ETC=0)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

OPTION(BUILD_BENCHMARKS "samurai benchmark suite" OFF)
OPTION(BUILD_DEMOS "samurai build all demos" OFF)
OPTION(BUILD_TESTS "samurai test suite" OFF)
OPTION(WITH_STATS "samurai mesh stats" OFF)
option(SAMURAI_CHECK_NAN "Check NaN in computations" OFF)

if(WITH_STATS)
  find_package(nlohmann_json REQUIRED)
  target_link_libraries(samurai INTERFACE nlohmann_json::nlohmann_json)
  target_compile_definitions(samurai INTERFACE WITH_STATS)
endif()

if(SAMURAI_CHECK_NAN)
  target_compile_definitions(samurai INTERFACE SAMURAI_CHECK_NAN)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(BUILD_DEMOS)
  add_subdirectory(demos)
else()
  add_subdirectory(demos EXCLUDE_FROM_ALL)
endif()

if(NOT FLUX_CONTAINER IN_LIST FLUX_CONTAINER_LIST)
    message(FATAL_ERROR "FLUX_CONTAINER must be one of: ${FLUX_CONTAINER_LIST}")
else()
    target_compile_definitions(samurai INTERFACE FLUX_CONTAINER_${FLUX_CONTAINER})
endif()

# Package the project
package_project(
  TARGETS samurai project_options project_warnings libdeps
  INTERFACE_DEPENDENCIES_CONFIGURED ${DEPENDENCIES_CONFIGURED}
  INTERFACE_INCLUDES ${INCLUDE_DIR}
)
