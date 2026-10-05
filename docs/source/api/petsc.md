# PETSc solvers

This page documents the PETSc solvers of {{ project }}, which solve the implicit equation $D(x) = b$ for a discrete operator $D$ built as in {doc}`schemes`.
`samurai::petsc::solve(D, x, b)` builds a solver and solves once; `samurai::petsc::make_solver(D)` returns the solver, so that you can configure it and solve several times with the matrix assembled once.
`make_solver` picks the solver class from the scheme type: `samurai::petsc::LinearSolver` for a linear scheme, `samurai::petsc::NonLinearSolver` (a PETSc SNES Newton solver) for a non-linear one, `samurai::petsc::NonLinearLocalSolvers` for a non-linear cell-based scheme with a stencil of one cell, and the block solvers for a block operator.
The solvers are available when the program is compiled with `SAMURAI_WITH_PETSC` defined; include `samurai/schemes/fv.hpp`.
For examples, see the {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` and the {doc}`local schemes reference <../reference/local_schemes>`.

## Solve and create a solver

`solve(D, x, b)` throws `std::runtime_error` when the solver diverges.
A flux-based scheme needs boundary conditions attached to the unknown `x`.
For a block operator, `make_solver<BlockAssemblyType::NestedMatrices>(block_operator)` assembles one PETSc matrix per block instead of one monolithic matrix; `make_nonlinear_solver` uses the non-linear block solver for a linear block operator.

```{doxygenfile} solver_helpers.hpp
:sections: innernamespace func
```

```{doxygenenum} samurai::petsc::BlockAssemblyType
```

## Linear solvers

`LinearSolver` assembles the matrix of the scheme at the first call to `solve` (or to `setup`) and solves with a PETSc KSP.
The callbacks `configure`, `after_matrix_assembly` and `after_setup` give access to the KSP, its preconditioner and the matrix; `configure` runs before `KSPSetFromOptions`, so the PETSc command-line options take precedence.

```{doxygenclass} samurai::petsc::LinearSolver
:members: LinearSolver, set_unknown, solve
:undoc-members:
```

```{doxygenclass} samurai::petsc::LinearSolverBase
:members: configure, after_matrix_assembly, after_setup, Ksp, setup, iterations, reset, set_scheme
:undoc-members:
```

## Non-linear solvers

`NonLinearSolver` solves with a PETSc SNES, from the initial guess held by the unknown.
When the SNES diverges, it throws `std::runtime_error`, unless `stop_program_on_divergence(false)` was called.
`monitor`, when set, is called at each Newton iteration with the iteration number, the residual norm and the residual field.

```{doxygenclass} samurai::petsc::NonLinearSolver
:members: NonLinearSolver, set_unknown, solve
:undoc-members:
```

```{doxygenclass} samurai::petsc::NonLinearSolverBase
:members: configure, after_matrix_assembly, monitor, Snes, stop_program_on_divergence, setup, iterations, reset, set_scheme
:undoc-members:
```

`NonLinearLocalSolvers` solves a non-linear local scheme with one independent SNES per cell, of the size of the number of field components.

```{doxygenclass} samurai::petsc::NonLinearLocalSolvers
:members: NonLinearLocalSolvers, configure, set_unknown, solve, setup, reset
:undoc-members:
```

## Block solvers

The block solvers solve a system built by `samurai::make_block_operator`.
`set_unknowns(u_1, ..., u_n)` takes one unknown field per block column, and `solve(b_1, ..., b_n)` one right-hand side field per block row.

```{doxygenclass} samurai::petsc::LinearBlockSolver
:members: LinearBlockSolver, set_unknowns, set_unknown, solve, set_pc_fieldsplit
:undoc-members:
```

```{doxygenclass} samurai::petsc::NonLinearBlockSolver
:members: NonLinearBlockSolver, set_unknowns, set_unknown, solve
:undoc-members:
```
