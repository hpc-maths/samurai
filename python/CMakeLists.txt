# Python Bindings with pybind11
cmake_minimum_required(VERSION 3.16)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Find pybind11 (use FetchContent if not installed)
include(FetchContent)

# Try to find pybind11 locally first
find_package(pybind11 2.10 CONFIG)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "pybind11 fetched and configured")
else()
    message(STATUS "pybind11 found: ${pybind11_VERSION}")
endif()

# Create Python module
pybind11_add_module(samurai_python
    src/bindings/main.cpp
    src/bindings/box_bindings.cpp
    src/bindings/mesh_config_bindings.cpp
    src/bindings/mesh_bindings.cpp
    src/bindings/field_bindings.cpp
    src/bindings/interval_bindings.cpp
    src/bindings/algorithm_bindings.cpp
    src/bindings/operator_bindings.cpp
)

# Set target properties
set_target_properties(samurai_python PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Link against Samurai (interface library)
target_link_libraries(samurai_python
    PRIVATE
        samurai
)

# Include directories
target_include_directories(samurai_python
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${Python_INCLUDE_DIRS}
)

# C++ standard
target_compile_features(samurai_python PRIVATE cxx_std_20)

# Export module symbol for Windows
if(WIN32)
    target_compile_definitions(samurai_python PRIVATE "WIN32_LEAN_AND_MEAN")
endif()

# Output directory
set_target_properties(samurai_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# Optional: Create test target
if(BUILD_TESTING)
    enable_testing()
    add_test(NAME python_import
        COMMAND ${Python_EXECUTABLE} -c "import sys; sys.path.insert(0, '${CMAKE_BINARY_DIR}/python'); import samurai"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS samurai_python
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Python Bindings Configuration ===")
message(STATUS "  Python executable: ${Python_EXECUTABLE}")
message(STATUS "  Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "  pybind11 version: ${pybind11_VERSION}")
message(STATUS "  Module output: ${CMAKE_BINARY_DIR}/python")
message(STATUS "=====================================")
message(STATUS "")
