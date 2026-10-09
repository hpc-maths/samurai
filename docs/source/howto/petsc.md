# How-to: solve an implicit scheme with PETSc

This guide shows you how to solve the system of an implicit scheme with [PETSc](https://petsc.org), sequentially or with MPI.
You build a solver from a discrete operator with `samurai::petsc::make_solver`, configure it in the code or from the command line, solve, and check that the solve converged.
The same steps cover linear schemes, non-linear schemes (Newton's method) and block systems of several unknown fields.

## Before you start

- PETSc and `pkg-config` are installed, as the {doc}`installation guide <installation>` explains.
- Your program is built with PETSc support:
  - in your own CMake project, set `SAMURAI_WITH_PETSC` to `ON` before `find_package(samurai)`, as the {doc}`CMake guide <cmake>` explains. It defines the macro `SAMURAI_WITH_PETSC` and links PETSc, found through `pkg-config`;
  - in the samurai source tree, `-DWITH_PETSC=ON` builds the demos and tests that use PETSc.

  Run your program with `--info`: the `Build configuration` section must show `PETSc : ON`.
  Without PETSc support, the namespace `samurai::petsc` does not exist and the solver calls fail to compile.
- You include `samurai/schemes/fv.hpp` and have a discrete operator to invert: a built-in one from the {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` or the {doc}`local schemes reference <../reference/local_schemes>`, or one you wrote as in the {doc}`finite volume scheme how-to guide <fv_scheme>`.

The examples of this guide live in `docs/source/howto/snippet/petsc/`.
Each one solves three backward Euler steps of a heat equation on a uniform 2D mesh of the unit square, and prints the iterations and the residual of each solve.
In the samurai source tree, `-DBUILD_SNIPPETS=ON` builds them when `-DWITH_PETSC=ON` is set too.

## Pass the command line to PETSc

Call `samurai::initialize` with `argc` and `argv`, then `SAMURAI_PARSE`, at the start of `main`, and `samurai::finalize` at the end:

```{literalinclude} snippet/petsc/petsc_linear_heat.cpp
  :language: c++
  :start-at: int main
```

In a PETSc build, `samurai::initialize` calls `PetscInitialize` with `argc` and `argv`, which also initializes MPI, and `samurai::finalize` calls `PetscFinalize`.
samurai ignores the options it does not know, so the PETSc options (one leading dash, as in `-ksp_type`) reach PETSc and the samurai options (two dashes, as in `--max-level`) reach samurai.
The overload `samurai::initialize()` without arguments does not initialize PETSc: do not use it with PETSc.

Create the solvers in a function or a block that ends before `samurai::finalize`, as `solve_heat` does above.
A solver frees its PETSc objects when it is destroyed, and freeing them after `samurai::finalize` aborts the program, with Open MPI on the message `The MPI_Comm_get_attr() function was called after MPI_FINALIZE was invoked.`

If the solver must live in the same scope as `samurai::finalize`, call `solver.destroy_petsc_objects()` before `samurai::finalize`, as the demos in `demos/FiniteVolume/` do.

## Solve a linear scheme

The function `solve_heat` solves $u_{n+1} + \Delta t\, \mathcal{D}(u_{n+1}) = u_n$ at each step, where $\mathcal{D}$ is the diffusion operator:

```{literalinclude} snippet/petsc/petsc_linear_heat.cpp
  :language: c++
  :start-at: void solve_heat
  :end-before: int main
```

1. Attach boundary conditions to the unknown field, here `unp1`, as the {doc}`boundary conditions how-to guide <bc>` explains.
   The solver reads them to write the equations of the ghosts outside the domain; it handles `samurai::Dirichlet` and `samurai::Neumann` only.
   The right-hand side `u` needs none.
2. Build the operator to invert, here `A = id + dt * diff`.
   It is not a matrix: the solver assembles the PETSc matrix from it.
3. Create the solver with `samurai::petsc::make_solver(A)`.
   For a linear operator, it returns a `samurai::petsc::LinearSolver`, which solves with a PETSc KSP.
4. Give the solver the unknown field with `set_unknown`.
   The solver keeps a reference to it and writes the solution into it.
5. Call `solve(rhs)`.
   The first call assembles the matrix and sets up the KSP; the next calls reuse them.
   `solve(unp1, u)` sets the unknown and solves in one call.

For a single solve, `samurai::petsc::solve(A, unp1, u)` creates a solver, solves and destroys the solver.
It assembles the matrix at every call, so keep a solver object when you solve the same system several times.

## Configure the solver

Set the default solver of your program in the `configure` callback, as `solve_heat` does with a direct LU solve:

```{literalinclude} snippet/petsc/petsc_linear_heat.cpp
  :language: c++
  :start-at: solver.configure
  :end-at: };
  :dedent:
```

The callback receives the KSP and its preconditioner (PC), and runs before `KSPSetFromOptions`.
The options of the command line therefore replace the choices made in `configure`.
Without `configure` and without options, the PETSc defaults apply: GMRES with an ILU preconditioner on one process, and block Jacobi on several (see the [PETSc KSP manual](https://petsc.org/release/manual/ksp/)).

To switch to GMRES with a Jacobi preconditioner without recompiling, run:

```bash
./petsc_linear_heat -ksp_type gmres -pc_type jacobi \
                    -ksp_converged_reason
```

```{literalinclude} snippet/petsc/petsc_linear_heat_gmres_output.txt
  :language: text
```

The digits of the residual depend on the BLAS library that PETSc uses, so the output shows them as `x`.

The options you will use most:

| Option | Effect |
| --- | --- |
| `-ksp_type <type>` | Krylov method: `gmres`, `cg` (symmetric positive definite matrices), `preonly` (apply the preconditioner once, for a direct solve). |
| `-pc_type <type>` | Preconditioner: `lu`, `ilu`, `jacobi`, `bjacobi`, `fieldsplit`... |
| `-ksp_rtol <tol>` | Relative tolerance of the Krylov method, `1e-5` by default. |
| `-ksp_max_it <n>` | Maximum number of iterations. |
| `-ksp_monitor` | Prints the preconditioned residual norm at each iteration. |
| `-ksp_monitor_true_residual` | Prints the preconditioned and the unpreconditioned residual norms at each iteration. |
| `-ksp_converged_reason` | Prints why each solve stopped and after how many iterations. |
| `-ksp_view` | Prints the full configuration of the solver and of its matrix. |
| `-help` | Prints the PETSc options of the objects the program creates. |

The non-linear solver takes the same options for the linear solve of each Newton iteration, and the `-snes_` options listed in the [next section](#solve-a-non-linear-scheme).
The [PETSc manual](https://petsc.org/release/manual/) describes them all.

samurai turns PETSc's `-options_left` check off, so PETSc does not warn about a misspelled option: `-ksp_tpye gmres` is ignored without a message.
Run once with `-ksp_view` (or `-snes_view`) to check that the solver uses the configuration you asked for.

The solver also gives access to the PETSc objects from the code:

- `solver.Ksp()` returns the KSP (`solver.Snes()` returns the SNES of a non-linear solver);
- `after_matrix_assembly` is called with the KSP, the PC and the matrix once the matrix is assembled;
- `after_setup` is called once the KSP is set up, for settings that need the preconditioner built, such as the sub-solvers of a Schur complement in `demos/FiniteVolume/stokes_2d.cpp`.

## Solve a non-linear scheme

For an operator built from a `SchemeType::NonLinear` scheme, `samurai::petsc::make_solver` returns a `samurai::petsc::NonLinearSolver`, which runs Newton's method with a PETSc SNES.
The example `petsc_nonlinear_heat.cpp` replaces the diffusion with the non-linear flux $u \nabla u$ of `demos/FiniteVolume/heat_nonlinear.cpp`:

```{literalinclude} snippet/petsc/petsc_nonlinear_heat.cpp
  :language: c++
  :start-at: auto u    =
  :end-at: // end of the time loop
  :dedent:
```

The differences with the linear case are:

- Newton's method starts from the values of the unknown field: set them before each `solve`, here with `unp1 = u`.
- The field assignment `unp1 = u` copies the boundary conditions of `u` along with its values, and replaces those of `unp1`.
  Attach the same boundary conditions to both fields, as the example does, or set the initial guess cell by cell.
- The `configure` callback receives the SNES, the KSP of the Newton iterations and its PC.
- The scheme needs a Jacobian function.
  Without one, the assembly throws `std::runtime_error`; the options `-snes_mf` and `-snes_fd` compute the Jacobian instead, as the {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` explains.
- `solve(rhs)` returns the PETSc `SNESConvergedReason`, negative when Newton's method diverged.

Follow the Newton iterations with `-snes_monitor`:

```bash
./petsc_nonlinear_heat -snes_monitor
```

```{literalinclude} snippet/petsc/petsc_nonlinear_heat_monitor_output.txt
  :language: text
```

PETSc prints the norms with 13 digits; the output above keeps 6, the ones that do not depend on the processor.

The residual norm drops quadratically, the sign of a correct Jacobian.
The most used SNES options are `-snes_type` (`newtonls` by default), `-snes_rtol`, `-snes_atol`, `-snes_max_it`, `-snes_monitor`, `-snes_converged_reason` and `-snes_view`.
To receive the residual as a samurai field instead, set the `monitor` callback of the solver: it is called at each Newton iteration with the iteration number, the residual norm and the residual field.

For a non-linear local scheme (a `LocalCellSchemeConfig` scheme), `make_solver` returns a `samurai::petsc::NonLinearLocalSolvers`, which runs one small Newton solve per cell, as the {doc}`local schemes reference <../reference/local_schemes>` explains.
On a scalar field, this solve does not compile ([#581](https://github.com/hpc-maths/samurai/issues/581)): use a vector field with one component, as `demos/FiniteVolume/nagumo.cpp` does.

## Solve a block system

When the scheme couples several unknown fields, build a block operator with `samurai::make_block_operator<rows, cols>`, giving the operators row by row.
The example `petsc_block_system.cpp` couples two heat equations through a linear exchange term:

```{literalinclude} snippet/petsc/petsc_block_system.cpp
  :language: c++
  :start-at: // u' = lap(u)
  :end-at: // end of the time loop
  :dedent:
```

- `set_unknowns` takes one unknown field per block column, in order.
- `solve` takes one right-hand side field per block row.
- `samurai::petsc::make_solver(op)` assembles one monolithic PETSc matrix.
  `make_solver<NestedMatrices>(op)`, with `NestedMatrices` from `samurai::petsc::BlockAssemblyType`, assembles one PETSc matrix per block instead.
- `set_pc_fieldsplit(pc)` turns the preconditioner into a PETSc field split with one split per unknown field.
  Call it in `after_matrix_assembly`, because a nested matrix must be assembled first.
  Each split takes the name of its unknown field, here `u` and `v`, which gives the option prefixes `-fieldsplit_u_` and `-fieldsplit_v_`:

```bash
./petsc_block_system -ksp_converged_reason \
                     -fieldsplit_u_pc_type lu -fieldsplit_v_pc_type lu
```

```{literalinclude} snippet/petsc/petsc_block_system_output.txt
  :language: text
```

With the default sub-solvers (ILU on each split), each step takes 7 or 8 iterations.
`demos/FiniteVolume/stokes_2d.cpp` configures a Schur complement field split for the Stokes system.

If one of the blocks is non-linear, `make_solver` returns a `samurai::petsc::NonLinearBlockSolver`, which runs Newton's method on the whole system, as in `demos/FiniteVolume/diff_heated_cavity.cpp`.
`samurai::petsc::make_nonlinear_solver(op)` uses the non-linear block solver for a linear block operator.

## Reuse the solver in a time loop

A solver keeps the matrix it assembled, which is right as long as the operator and the mesh do not change.
When one of them changes, tell the solver, as `demos/FiniteVolume/heat.cpp` does:

- When the operator changes, for example because the last time step is shortened and the operator contains `dt`, pass the new operator with `set_scheme`:

  ```{literalinclude} ../../../demos/FiniteVolume/heat.cpp
    :language: c++
    :start-at: if (t > Tf)
    :end-before: std::cout << fmt::format("iteration
    :dedent:
  ```

- When the mesh changes after an adaptation, call `reset` so that the solver assembles a new matrix on the new mesh:

  ```{literalinclude} ../../../demos/FiniteVolume/heat.cpp
    :language: c++
    :start-after: // Mesh adaptation
    :end-before: if (explicit_scheme)
    :dedent:
  ```

  The unknown field must be resized to the new mesh too, here with `unp1.resize()`.

## Check the result

A solve that fails throws `std::runtime_error` from `solve`, with the PETSc reason:

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  Divergence of the solver (DIVERGED_ITS)
```

The non-linear solvers throw `Divergence of the non-linear solver (...)` in the same way.
To handle the failure yourself, for example to retry with a smaller time step, call `solver.stop_program_on_divergence(false)` on a non-linear solver and test the reason that `solve` returns, as `petsc_nonlinear_heat.cpp` does:

```bash
./petsc_nonlinear_heat -snes_max_it 1
```

```{literalinclude} snippet/petsc/petsc_nonlinear_heat_diverged_output.txt
  :language: text
```

A solve that does not throw has met the PETSc stopping criterion, but that criterion may not be the one you need.
By default, the Krylov methods stop when the preconditioned residual norm of the whole system, ghost rows included, has dropped by a factor `1e-5`.
To check the solution of your scheme, compute the residual $b - A(x)$ on the cells, with the explicit application of the operator:

```{literalinclude} snippet/petsc/petsc_linear_heat.cpp
  :language: c++
  :start-at: auto Au
  :end-at: double r
  :dedent:
```

`max_diff` returns the largest $|b - A(x)|$ over the cells of all MPI ranks:

```{literalinclude} snippet/petsc/petsc_linear_heat.cpp
  :language: c++
  :start-at: template <class Field>
  :end-before: void solve_heat
```

For `petsc_linear_heat`, with $u_n$ of order 1:

| Options | Iterations per step | Residual on the cells |
| --- | --- | --- |
| none (`configure` sets LU) | 1 | $1.5 \times 10^{-14}$ to $1.9 \times 10^{-14}$ |
| `-ksp_type gmres -pc_type jacobi` | 14 to 22 | $2.6 \times 10^{-6}$ to $3.7 \times 10^{-6}$ |
| `-ksp_type gmres -pc_type jacobi -ksp_rtol 1e-10` | 48 to 56 | $2.6 \times 10^{-11}$ to $3.4 \times 10^{-11}$ |

If the residual is larger than the accuracy you need, lower `-ksp_rtol` or use a stronger preconditioner.
`solver.iterations()` returns the number of iterations of the last solve (Newton iterations for a non-linear solver).

## If you run with MPI

Build with `SAMURAI_WITH_MPI` and `SAMURAI_WITH_PETSC`, with a PETSc built on the same MPI library as samurai.
The code does not change: every rank creates the solver and calls `solve`, which are collective.
PETSc distributes the matrix rows by the cells each rank owns.

```bash
mpiexec -n 2 ./petsc_linear_heat -ksp_converged_reason
```

```{literalinclude} snippet/petsc/petsc_linear_heat_np2_output.txt
  :language: text
```

On several processes:

- Only rank 0 prints: samurai sends the standard output of the other ranks to `/dev/null`, unless you pass `--dont-redirect-output`.
- PETSc's own LU factorization runs on one process only.
  On several processes, `-pc_type lu` uses an external package: with the conda-forge PETSc, `-ksp_view` shows `package used to perform factorization: mumps`.
  Choose the package with `-pc_factor_mat_solver_type`.
- Preconditioners such as block Jacobi depend on the partition, so the iteration counts change with the number of processes: `-ksp_type gmres -pc_type bjacobi` takes 12 iterations per step on 2 processes.
- For `set_pc_fieldsplit`, use `NestedMatrices`, as in `petsc_block_system.cpp`.
  With a monolithic block matrix on 2 processes, the field split preconditioner fails with `Matrix is missing diagonal entries`.

To see how the unknowns are distributed, pass `--print-petsc-numbering`.
It has an effect in MPI builds only.
Each rank prints, for each cell it holds (ghosts included), the rank that owns the cell and its local and global PETSc indices, and the program writes the fields `owner_rank`, `samurai_cell_index` and `petsc_cell_index` to `petsc_indices.h5` and `petsc_indices.xdmf` in the current directory.
The ranks pause for a number of seconds equal to their rank before printing, so that their outputs do not mix.
On a 4 by 4 mesh:

```bash
mpiexec -n 2 ./petsc_linear_heat --min-level 2 --max-level 2 \
             --print-petsc-numbering --dont-redirect-output
```

```text
[0]: Cell ownership: owned: 32, total: 48
...
[0]: n_owned_unknowns = 32, n_ghost_unknowns = 16
[0]: OWNED local_index = [0,31], global_index = [0,31]
rank 0: GHOSTS local_index = [32,47]
[1]: n_owned_unknowns = 32, n_ghost_unknowns = 16
[1]: OWNED local_index = [0,31], global_index = [32,63]
rank 1: GHOSTS local_index = [32,47]
...
```

Rank 0 owns the global rows [0, 32) and rank 1 the rows [32, 64); each rank also holds 16 unknowns owned by the other.

The 4 by 4 mesh has 16 cells, yet the system has 64 unknowns, because the boundary ghosts are unknowns too.
The mesh keeps two layers of ghosts around the domain, and the solver gives each of them an equation: the boundary condition for the ghosts that touch a face of the domain, and $x = 0$ (1 on the diagonal, 0 in the right-hand side) for the others, in the outer layer and in the corners.
The 8 by 8 cells and ghosts are numbered row by row, and each rank owns half of them, as {ref}`plate-petsc-numbering` shows.

```{plate} The PETSc numbering on two ranks
:figure: petsc_numbering
:label: plate-petsc-numbering
:columns: 1

Each square is an unknown of the system, with its global PETSc index, read from `petsc_indices.h5`.
The inner heavy square is the domain, 4 by 4 cells of level 2; the two layers around it are the boundary ghosts.

Rank 0 owns the hatched rows and rank 1 the dotted rows, on either side of the heavy rule.
Each rank owns 8 cells of the domain and the 24 ghosts beside them, 32 unknowns.
```

## Related

- The {doc}`reaction-diffusion tutorial <../tutorial/reaction_diffusion>`, which solves linear and non-linear implicit schemes step by step.
- The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` and the {doc}`local schemes reference <../reference/local_schemes>`, to build the operators you solve, and {doc}`fv_scheme`, to write your own.
- The {doc}`PETSc API reference <../api/petsc>`, for the solver classes and their members.
- {doc}`cmake`, to turn on PETSc and MPI in your project, and {doc}`options`, for the samurai command-line options.
- The demos `demos/FiniteVolume/heat.cpp`, `heat_nonlinear.cpp`, `nagumo.cpp`, `stokes_2d.cpp` and `lid_driven_cavity.cpp`.
