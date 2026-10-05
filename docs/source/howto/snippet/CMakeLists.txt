file(GLOB snippet_list "**/*.cpp")
# The PETSc snippets are built below, only when PETSc is enabled
list(FILTER snippet_list EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/petsc/")

foreach(snippet_file ${snippet_list})
    get_filename_component(snippet_name ${snippet_file} NAME_WE)
    add_executable(${snippet_name} ${snippet_file})
     target_link_libraries(${snippet_name} PRIVATE samurai)
endforeach()

if(${WITH_PETSC})
    include(CMakeFindDependencyMacro)
    find_dependency(PkgConfig)
    pkg_check_modules(PETSC REQUIRED PETSc)

    include_directories(${PETSC_INCLUDE_DIRS})

    file(GLOB petsc_snippet_list "petsc/*.cpp")
    foreach(snippet_file ${petsc_snippet_list})
        get_filename_component(snippet_name ${snippet_file} NAME_WE)
        add_executable(${snippet_name} ${snippet_file})
        target_compile_definitions(${snippet_name} PUBLIC SAMURAI_WITH_PETSC)
        target_link_libraries(${snippet_name} samurai ${PETSC_LINK_LIBRARIES})
    endforeach()
endif()
