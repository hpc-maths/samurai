file(GLOB snippet_list "*/*.cpp")

foreach(snippet_file ${snippet_list})
    get_filename_component(snippet_name ${snippet_file} NAME_WE)
    add_executable(${snippet_name} ${snippet_file})
    target_link_libraries(${snippet_name} PRIVATE samurai)
endforeach()

# Outputs shown in the tutorials (see cmake/snippetOutputs.cmake)
samurai_snippet_output(interval_1d)
samurai_snippet_output(2d_mesh_box)
samurai_snippet_output(burgers_1d TAIL 1)
