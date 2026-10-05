file(GLOB snippet_list "*/*.cpp")

foreach(snippet_file ${snippet_list})
    get_filename_component(snippet_name ${snippet_file} NAME_WE)
    add_executable(${snippet_name} ${snippet_file})
    target_link_libraries(${snippet_name} PRIVATE samurai)
endforeach()

# Outputs shown in the reference pages (see cmake/snippetOutputs.cmake)
samurai_snippet_output(set_algebra_boolean)
samurai_snippet_output(set_algebra_expand_contract)
samurai_snippet_output(set_algebra_levels)
samurai_snippet_output(set_algebra_dynamic)
