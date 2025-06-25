add_subdirectory(AMR_1D_Burgers)

# List of tutorial demos with their corresponding executable names
set(TUTORIAL_DEMOS
    2D_mesh.cpp:tutorial-2d-mesh
    interval.cpp:tutorial-interval
    set_operator.cpp:tutorial-set-operator
    graduation_case_1.cpp:tutorial-graduation-case-1
    graduation_case_2.cpp:tutorial-graduation-case-2
    graduation_case_3.cpp:tutorial-graduation-case-3
    reconstruction_1d.cpp:tutorial-reconstruction-1d
    reconstruction_2d.cpp:tutorial-reconstruction-2d
    reconstruction_3d.cpp:tutorial-reconstruction-3d
    proj_on_mesh.cpp:tutorial-proj-on-mesh
)

# Create executables for all tutorial demos
foreach(demo_entry ${TUTORIAL_DEMOS})
    string(REPLACE ":" ";" demo_parts ${demo_entry})
    list(GET demo_parts 0 source_file)
    list(GET demo_parts 1 executable_name)

    add_executable(${executable_name} ${source_file})
    target_link_libraries(${executable_name} PRIVATE samurai CLI11::CLI11)
endforeach()

# Specific options for MSVC
if(MSVC)
    target_compile_options(tutorial-reconstruction-3d PUBLIC /bigobj)
endif()
