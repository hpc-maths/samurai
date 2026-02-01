if(${WITH_PETSC})
    include(CMakeFindDependencyMacro)
    find_dependency(PkgConfig)
    pkg_check_modules(PETSC REQUIRED PETSc)

    include_directories(${PETSC_INCLUDE_DIRS})

    add_executable(multigrid main.cpp)
    target_compile_definitions(multigrid PUBLIC SAMURAI_WITH_PETSC)
    target_link_libraries(multigrid samurai ${PETSC_LINK_LIBRARIES})

    # Link pre-instantiated templates if enabled
    if(SAMURAI_INSTANTIATE_TEMPLATES)
        target_link_libraries(multigrid samurai_instantiations)
    endif()
endif()
