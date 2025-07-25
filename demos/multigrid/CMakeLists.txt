if(${WITH_PETSC})
    include(CMakeFindDependencyMacro)
    find_dependency(PkgConfig)
    pkg_check_modules(PETSC REQUIRED PETSc)

    include_directories(${PETSC_INCLUDE_DIRS})

    add_executable(multigrid main.cpp)
    target_compile_definitions(multigrid PUBLIC SAMURAI_WITH_PETSC)
    target_link_libraries(multigrid samurai CLI11::CLI11 ${PETSC_LINK_LIBRARIES})
endif()
