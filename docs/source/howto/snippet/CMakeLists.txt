file(GLOB snippet_list "**/*.cpp")
# The PETSc snippets are built below, only when PETSc is enabled
list(FILTER snippet_list EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/petsc/")

foreach(snippet_file ${snippet_list})
    get_filename_component(snippet_name ${snippet_file} NAME_WE)
    add_executable(${snippet_name} ${snippet_file})
     target_link_libraries(${snippet_name} PRIVATE samurai)
endforeach()

# Outputs shown in the how-to guides (see cmake/snippetOutputs.cmake)
samurai_snippet_output(mr_adapt)
samurai_snippet_output(mr_adapt OUTPUT mr_adapt_eps_output.txt ARGS --mr-eps 1e-4)
samurai_snippet_output(init_constant)
samurai_snippet_output(init_function)
samurai_snippet_output(init_gauss_legendre)
samurai_snippet_output(two_holes)
samurai_snippet_output(disc_obstacle)
samurai_snippet_output(for_each_cell_mesh HEAD 2 TAIL 1)
samurai_snippet_output(for_each_interval_mesh HEAD 2 TAIL 1)
samurai_snippet_output(uniform)
samurai_snippet_output(mrmesh)
samurai_snippet_output(amrmesh)
samurai_snippet_output(ghost_config)
samurai_snippet_output(predefined_options OUTPUT predefined_options_help_output.txt ARGS --help HEAD 13)
samurai_snippet_output(predefined_options ARGS --max-level 7)
samurai_snippet_output(disable_args_parse ARGS --max-level 7)
samurai_snippet_output(custom_options ARGS --my-option 3 --my-flag)
samurai_snippet_output(custom_options OUTPUT custom_options_config_output.txt ARGS --config config.toml INPUTS config.toml)
samurai_snippet_output(restart_mesh_fields)
samurai_snippet_output(bc_dirichlet_neumann)
samurai_snippet_output(bc_regions)
samurai_snippet_output(bc_periodic)
samurai_snippet_output(bc_custom)
samurai_snippet_output(fv_laplacian)
samurai_snippet_output(fv_gradient)
samurai_snippet_output(fv_upwind)
# The gap between the Jacobian and its finite differences is round-off: its
# digits change with the processor, keep only its exponent
samurai_snippet_output(fv_burgers
    REPLACE [[(finite differences: )\d\.\d+e]] [[\1x.xxe]])
samurai_snippet_output(mpi_adapt NP 1 OUTPUT mpi_adapt_np1_output.txt)
samurai_snippet_output(mpi_adapt NP 2 OUTPUT mpi_adapt_np2_output.txt)
samurai_snippet_output(mpi_adapt NP 4 OUTPUT mpi_adapt_np4_output.txt)
samurai_snippet_output(mpi_adapt NP 4 OUTPUT mpi_adapt_load_balancing_output.txt ARGS --load-balancing-at 1)

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

    # The digits of these residuals change with the BLAS library: keep
    # only their exponent
    set(residual_digits [[residual \d(?:\.\d+)?e([-+]\d+)$]] [[residual x.xxe\1]])
    samurai_snippet_output(petsc_linear_heat OUTPUT petsc_linear_heat_gmres_output.txt
        ARGS -ksp_type gmres -pc_type jacobi -ksp_converged_reason
        REPLACE ${residual_digits})
    # PETSc prints the SNES norms with 13 digits; the last ones change with
    # the processor (aarch64 and x86_64 differ at the 11th): keep 6
    samurai_snippet_output(petsc_nonlinear_heat OUTPUT petsc_nonlinear_heat_monitor_output.txt
        ARGS -snes_monitor HEAD 5
        REPLACE [[(SNES Function norm \d\.\d{5})\d+(e[-+]\d+)$]] [[\1\2]])
    samurai_snippet_output(petsc_nonlinear_heat OUTPUT petsc_nonlinear_heat_diverged_output.txt
        ARGS -snes_max_it 1)
    samurai_snippet_output(petsc_block_system
        ARGS -ksp_converged_reason -fieldsplit_u_pc_type lu -fieldsplit_v_pc_type lu)
    samurai_snippet_output(petsc_linear_heat NP 2 OUTPUT petsc_linear_heat_np2_output.txt
        ARGS -ksp_converged_reason
        REPLACE ${residual_digits})
endif()
