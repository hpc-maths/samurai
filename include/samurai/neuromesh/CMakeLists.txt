# CMakeLists.txt for NeuroMesh

# Collect source files
set(NEUROMESH_SOURCES
    # Add any .cpp files here if we create them
)

set(NEUROMESH_HEADERS
    neuromesh.hpp
    README.md
)

# Create interface library for header-only
add_library(samurai_neuromesh INTERFACE)
target_include_directories(samurai_neuromesh
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(samurai_neuromesh
    INTERFACE
        samurai::core
        xtensor
)

# Install targets
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/samurai/neuromesh
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.md"
)

# Optional: Create demo
if(BUILD_DEMOS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../demos/neuromesh
                     ${CMAKE_CURRENT_BINARY_DIR}/neuromesh_demos)
endif()
