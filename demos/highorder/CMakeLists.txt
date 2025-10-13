if(${WITH_PETSC})
    include(CMakeFindDependencyMacro)
    find_dependency(PkgConfig)
    pkg_check_modules(PETSC REQUIRED PETSc)

    include_directories(${PETSC_INCLUDE_DIRS})

    add_executable(highorder main.cpp)
    target_compile_definitions(highorder PUBLIC SAMURAI_WITH_PETSC)
    target_link_libraries(highorder samurai ${PETSC_LINK_LIBRARIES})
endif()
