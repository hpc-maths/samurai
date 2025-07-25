cmake_minimum_required(VERSION 3.16)

# Set version
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" SAMURAI_VERSION)
  string(STRIP "${SAMURAI_VERSION}" SAMURAI_VERSION)
else()
  message(FATAL_ERROR "File ${CMAKE_CURRENT_SOURCE_DIR}/version.txt not found")
endif()

project(samurai VERSION ${SAMURAI_VERSION} LANGUAGES CXX C)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# Add project_options v0.41.0
# https://github.com/aminya/project_options
include(FetchContent)
FetchContent_Declare(_project_options URL https://github.com/aminya/project_options/archive/refs/tags/v0.41.0.zip)
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
option(WITH_PETSC "Enable PETSc" OFF)
option(SAMURAI_CONTAINER_LAYOUT_COL_MAJOR "Set the containers' layout to column-major" OFF)

set(FIELD_CONTAINER_LIST "xtensor" "eigen3")
set(SAMURAI_FIELD_CONTAINER "xtensor" CACHE STRING "Container to store fields: ${FIELD_CONTAINER_LIST}")
set_property(CACHE SAMURAI_FIELD_CONTAINER PROPERTY STRINGS ${FIELD_CONTAINER_LIST})

set(FLUX_CONTAINER_LIST "array" "xtensor" "eigen3")
set(SAMURAI_FLUX_CONTAINER "xtensor" CACHE STRING "Container to store fluxes: ${FLUX_CONTAINER_LIST}")
set_property(CACHE SAMURAI_FLUX_CONTAINER PROPERTY STRINGS ${FLUX_CONTAINER_LIST})

set(STATIC_MATRIX_CONTAINER_LIST "xtensor" "eigen3")
set(SAMURAI_STATIC_MAT_CONTAINER "xtensor" CACHE STRING "Container to store static matrices: ${STATIC_MATRIX_CONTAINER_LIST}")
set_property(CACHE SAMURAI_STATIC_MAT_CONTAINER PROPERTY STRINGS ${STATIC_MATRIX_CONTAINER_LIST})

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
target_compile_features(samurai INTERFACE cxx_std_20)
target_link_libraries(samurai INTERFACE project_options project_warnings)

# Includes
set(INCLUDE_DIR "include") # must be relative paths
target_include_directories(samurai INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Find dependencies:
set(DEPENDENCIES_CONFIGURED HighFive pugixml fmt)

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
  HighFive
  pugixml::pugixml
  fmt::fmt
)

if(SAMURAI_FIELD_CONTAINER MATCHES xtensor)
  message(STATUS "Using xtensor as field container")
  list(APPEND DEPENDENCIES_CONFIGURED xtensor)
  find_package(xtensor CONFIG REQUIRED)

  if (xtensor_VERSION VERSION_GREATER_EQUAL "0.26.0")
    message(FATAL_ERROR "xtensor version ${xtensor_VERSION} is too new. Maximum allowed version is 0.25.")
  endif()

  target_link_system_libraries(
    samurai
    INTERFACE
    xtensor
  )

endif()
if(SAMURAI_FIELD_CONTAINER MATCHES eigen3)
  message(STATUS "Using Eigen3 as field container")
endif()

if(SAMURAI_FIELD_CONTAINER MATCHES eigen3 OR SAMURAI_FLUX_CONTAINER MATCHES eigen3 OR SAMURAI_STATIC_MAT_CONTAINER MATCHES eigen3)
  message(STATUS "Using Eigen3 as field container")
  list(APPEND DEPENDENCIES_CONFIGURED Eigen3)
  find_package(Eigen3 CONFIG REQUIRED)
  target_link_system_libraries(
    samurai
    INTERFACE
    Eigen3::Eigen
  )
  target_compile_definitions(samurai INTERFACE EIGEN_ARRAYBASE_PLUGIN="${CMAKE_CURRENT_SOURCE_DIR}/include/samurai/storage/eigen/array_eigen_addons.hpp")
endif()

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
  if (NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "HDF5 is not parallel. Please install a parallel version.")
  endif()
  find_package(Boost REQUIRED COMPONENTS serialization mpi)
  target_link_system_libraries(
    samurai
    INTERFACE
    Boost::serialization
    Boost::mpi
  )
  target_compile_definitions(samurai INTERFACE SAMURAI_WITH_MPI)
endif()


target_compile_features(samurai INTERFACE cxx_std_20)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

OPTION(BUILD_BENCHMARKS "samurai benchmark suite" OFF)
OPTION(BUILD_DEMOS "samurai build all demos" OFF)
OPTION(BUILD_TESTS "samurai test suite" OFF)
OPTION(SPLIT_TESTS "samurai split each test and create an executable for each" OFF)
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

if(SAMURAI_CONTAINER_LAYOUT_COL_MAJOR)
  target_compile_definitions(samurai INTERFACE SAMURAI_CONTAINER_LAYOUT_COL_MAJOR)
endif()

if(NOT SAMURAI_FIELD_CONTAINER IN_LIST FIELD_CONTAINER_LIST)
  message(FATAL_ERROR "SAMURAI_FIELD_CONTAINER must be one of: ${FIELD_CONTAINER_LIST}")
else()
  string(TOUPPER ${SAMURAI_FIELD_CONTAINER} SAMURAI_FIELD_CONTAINER)
  target_compile_definitions(samurai INTERFACE SAMURAI_FIELD_CONTAINER_${SAMURAI_FIELD_CONTAINER})
endif()

if(NOT SAMURAI_FLUX_CONTAINER IN_LIST FLUX_CONTAINER_LIST)
    message(FATAL_ERROR "SAMURAI_FLUX_CONTAINER must be one of: ${FLUX_CONTAINER_LIST}")
else()
    string(TOUPPER ${SAMURAI_FLUX_CONTAINER} SAMURAI_FLUX_CONTAINER)
    target_compile_definitions(samurai INTERFACE SAMURAI_FLUX_CONTAINER_${SAMURAI_FLUX_CONTAINER})
endif()

if(NOT SAMURAI_STATIC_MAT_CONTAINER IN_LIST STATIC_MATRIX_CONTAINER_LIST)
    message(FATAL_ERROR "SAMURAI_STATIC_MAT_CONTAINER must be one of: ${STATIC_MATRIX_CONTAINER_LIST}")
else()
    string(TOUPPER ${SAMURAI_STATIC_MAT_CONTAINER} SAMURAI_STATIC_MAT_CONTAINER)
    target_compile_definitions(samurai INTERFACE SAMURAI_STATIC_MAT_CONTAINER_${SAMURAI_STATIC_MAT_CONTAINER})
endif()

# Package the project
package_project(
  TARGETS samurai project_options project_warnings libdeps
  INTERFACE_DEPENDENCIES_CONFIGURED ${DEPENDENCIES_CONFIGURED}
  INTERFACE_INCLUDES ${INCLUDE_DIR}
)

# Append the content of the file cmake/samuraiConfig.cmake.in to the samurai.Config.cmake generated by the function `package_project`
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/cmake/samuraiConfig.cmake.in CONFIG_TO_APPEND)
file(APPEND "${CMAKE_BINARY_DIR}/CMakeFiles/samuraiConfig.cmake.install" ${CONFIG_TO_APPEND})
