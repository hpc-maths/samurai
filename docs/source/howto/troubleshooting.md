# How-to: fix common error messages

This page lists error messages that {{ project }} programs often print, with their cause and a fix.
Search the page for the text of your message.
Messages from the configuration and the build come first, then messages printed when the program runs.

Numbers, names and boxes in the messages below come from the example that produced them; yours differ.
Run-time errors raised as C++ exceptions end the program with `terminate called after throwing an instance of '<exception type>'` and the message after `what():`, unless your code catches them.

## CMake cannot find samurai

```text
CMake Error at CMakeLists.txt:4 (find_package):
  By not providing "Findsamurai.cmake" in CMAKE_MODULE_PATH this project has
  asked CMake to find a package configuration file provided by "samurai", but
  CMake did not find one.

  Could not find a package configuration file provided by "samurai" with any
  of the following names:

    samurai.cps
    samuraiConfig.cmake
    samurai-config.cmake

  Add the installation prefix of "samurai" to CMAKE_PREFIX_PATH or set
  "samurai_DIR" to a directory containing one of the above files.  If
  "samurai" provides a separate development package or SDK, be sure it has
  been installed.
```

**Cause**: `find_package(samurai)` searches the standard prefixes and the prefixes of the directories in `PATH`.
samurai is installed elsewhere, or it is installed in a conda environment that is not active in this shell.

**Fix**: if you installed samurai with conda, activate the environment (`conda activate <env>`) before you run `cmake`: its `bin` directory is then in `PATH` and CMake finds the package.
Otherwise, pass the installation prefix: `-DCMAKE_PREFIX_PATH=<samurai-install-prefix>`.
See the [CMake how-to guide](cmake.md).

## Parallel HDF5 is missing

```text
CMake Error at .../share/samurai/samuraiConfig.cmake:64 (message):
  HDF5 is not parallel.  Please install a parallel version.
```

When you build samurai from source with `-DWITH_MPI=ON`, the same message comes from `CMakeLists.txt:211`.

**Cause**: you turned on MPI (`SAMURAI_WITH_MPI=ON` in your project, `WITH_MPI=ON` for samurai itself), and the HDF5 that CMake found has no MPI support.
samurai writes its files with parallel HDF5 in an MPI build.

**Fix**: install an HDF5 built with MPI.
With conda, `conda install -c conda-forge 'hdf5=*=mpi_*'`, together with an MPI implementation and Boost.MPI.
If CMake still finds the sequential HDF5, point `CMAKE_PREFIX_PATH` to the parallel one.
See {ref}`install-with-conda` in the installation guide.

## xtensor is too old

```text
CMake Error at CMakeLists.txt:173 (message):
  xtensor version 0.25.0 is too old.  Minimum allowed version is 0.26.
```

**Cause**: you build samurai from source (its demos, tests or documentation snippets), and the xtensor found by CMake is 0.25 or older.
The check is in the `CMakeLists.txt` of samurai.

**Fix**: install a newer xtensor, for example `conda install -c conda-forge 'xtensor>=0.26'`, or point `CMAKE_PREFIX_PATH` to one.
See the {ref}`requirements <install-requirements>` in the installation guide.

## The samurai/petsc.hpp header

```text
.../include/samurai/petsc.hpp:7:2: error: #error "Please replace '#include <samurai/petsc.hpp>' with '#include <samurai/schemes/fv.hpp>' and activate the cmake option SAMURAI_WITH_PETSC by adding the line 'set(SAMURAI_WITH_PETSC On)' before the line 'find_package(samurai)' in your CMakeLists.txt file."
```

**Cause**: the header `samurai/petsc.hpp` only exists to stop the build with this message.
The PETSc solvers are declared by `samurai/schemes/fv.hpp` when samurai is configured with `SAMURAI_WITH_PETSC`.

**Fix**: include `<samurai/schemes/fv.hpp>` and configure your project with `-DSAMURAI_WITH_PETSC=ON`, or set the option before `find_package(samurai)`.
See [Turn on optional features](cmake.md#turn-on-optional-features) in the CMake guide.

## Mesh levels out of order

```text
terminate called after throwing an instance of 'std::invalid_argument'
  what():  Max level must be greater than min level.
```

or

```text
terminate called after throwing an instance of 'std::invalid_argument'
  what():  Start level 6 must lie between min level 3 and max level 5.
```

**Cause**: when the mesh is built, the max level is lower than the min level, or the start level lies outside the range from the min level to the max level.
The command-line options `--min-level`, `--max-level` and `--start-level` replace the values set in the code, so the values that fail can come from the command line.

**Fix**: pass levels in order on the command line, or set them in order in `samurai::mesh_config`.
To ignore the command line, call `disable_args_parse()` on the configuration.
See [Mesh options override the mesh configuration](options.md#mesh-options-override-the-mesh-configuration) in the options guide.

## A box that cannot be represented exactly

```text
The box Box({ 0.,  0.}, { 1.      ,  1.414214}) cannot be exactly represented with a reasonable cell length. You can modify the box's dimensions or you can set a tolerance so it can be approximately represented.
```

The program exits with status 1.

**Cause**: the mesh configuration calls `approx_box_tol(0)`, which asks for cells that fit the box exactly.
The side lengths of the box have no common divisor larger than $10^{-5}$ times a side, for example 1 and $\sqrt{2}$.

**Fix**: give the box side lengths with a simple ratio, or remove `approx_box_tol(0)`.
With the default tolerance of 0.05, samurai builds the mesh on a box close to yours.
See the [box how-to guide](box.md).

## A hole too small for the boundary conditions

```text
The hole Box({ 0.,  0.}, { 0.4,  0.4}) is too small to apply the BC at level 2 with the given scaling factor. We need to be able to construct 4 ghosts in each direction inside the hole.
Please choose a smaller scaling factor or enlarge the hole.
```

The program exits with status 1.

**Cause**: the mesh configuration sets `scaling_factor`, and at the min level a hole of the `DomainBuilder` is too small to hold the ghosts of the boundary conditions from both sides.

**Fix**: lower `scaling_factor`, raise the min level, or make the hole larger.
Without `scaling_factor`, samurai chooses a scaling factor that fits.
See the [troubleshooting section of the box guide](box.md#troubleshooting).

## A DomainBuilder with periodicity or MPI

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  Periodicity is not implemented with DomainBuilder.
```

or, in an MPI build:

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  MPI is not implemented with DomainBuilder.
```

**Cause**: `samurai::mra::make_mesh` received a `DomainBuilder` (a domain made of several boxes or with holes) and either a periodic mesh configuration or a build with `SAMURAI_WITH_MPI`.
Neither is supported with a `DomainBuilder`.

**Fix**: build the mesh from a single `samurai::Box` if you need periodicity or MPI, or run the program with a sequential build.
See [Build the mesh](box.md#build-the-mesh) in the box guide.

## Boundary condition stencil larger than the mesh allows

```text
terminate called after throwing an instance of 'std::invalid_argument'
  what():  The stencil size required by this boundary condition (6) is larger than the max_stencil_size parameter of the mesh (4).
You can set it with mesh_config.max_stencil_radius(3) or mesh_config.max_stencil_size(6).
```

**Cause**: `samurai::make_bc` attached a condition that fills more ghost layers than the mesh has, here `samurai::Dirichlet<3>` on a mesh with the default max stencil radius of 2.

**Fix**: set the max stencil radius in the mesh configuration to the value the message gives, `max_stencil_radius(3)`, or pass `--max-stencil-radius 3` on the command line.
See [Return value and errors](../reference/bc.md#return-value-and-errors) in the boundary conditions reference.

## Scheme stencil larger than the mesh allows

```text
terminate called after throwing an instance of 'std::invalid_argument'
  what():  The stencil size required by the scheme 'convection' (6) is larger than the max_stencil_size parameter of the mesh (4).
You can set it with mesh_config.max_stencil_radius(3) or mesh_config.max_stencil_size(6).
```

**Cause**: an explicit finite volume scheme reads further than the ghosts of the mesh reach, here the WENO5 convection scheme (stencil radius 3) on a mesh with the default max stencil radius of 2.

**Fix**: set the max stencil radius in the mesh configuration to the stencil radius of your widest scheme, `max_stencil_radius(3)` for WENO5, or pass `--max-stencil-radius 3` on the command line.
See [Errors](../reference/finite_volume_schemes.md#errors) in the finite volume schemes reference.

## Too many ghost layers for polynomial extrapolation

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  The outer ghosts are filled by polynomial extrapolation, which reaches 5 ghost layers at most, but the mesh has a ghost width of 6.
To fix this issue, lower the ghost width (it is half of mesh_config.max_stencil_size()), or raise max_stencil_size_implemented_PE in bc/polynomial_extrapolation.hpp and add the extrapolation coefficients that go with it.
```

**Cause**: the ghosts of a field were updated on a mesh whose max stencil radius is larger than 5.
The ghosts beyond those filled by the boundary condition are filled by polynomial extrapolation, which reaches 5 layers.

**Fix**: set the max stencil radius to 5 or less, in the mesh configuration or with `--max-stencil-radius`.
See [Ghosts beyond the condition](../reference/bc.md#ghosts-beyond-the-condition) in the boundary conditions reference.

## Reconstruction or transfer with too few ghosts

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  The reconstruction function requires at least 2 ghosts on the boundary.
To fix this issue, remove mesh_config.disable_minimal_ghost_width().
```

`samurai::transfer` prints the same message with `The transfer function`.

**Cause**: the mesh configuration calls `disable_minimal_ghost_width()` and keeps a max stencil radius of 1.
`samurai::reconstruction` and `samurai::transfer` predict values from two coarse cells and need two ghosts at the boundary.

**Fix**: remove `disable_minimal_ghost_width()`, or set `max_stencil_radius(2)` or more.
See [Set the ghost cells and the graduation](mesh.md#set-the-ghost-cells-and-the-graduation) in the mesh guide.

## A field interval that is not in the mesh

```text
terminate called after throwing an instance of 'std::out_of_range'
  what():  Field 'u' interval query failed on level 4: requested interval [10, 20)@0:1 could not be found for indices []; available interval: [-2, 18)@18:1
```

or, when no interval of the level contains the start of the requested one:

```text
terminate called after throwing an instance of 'std::out_of_range'
  what():  LevelCellArray::get_interval: interval not found at level 4, i = [30,40)@0:1, index =
```

**Cause**: an expression `u(level, i)` (or `u(level, i, j)` in 2D) asked for the values of `u` on an interval that the mesh does not hold at that level, ghosts included.
In the first message, the mesh holds `[-2, 18)` at level 4, which does not contain the whole of `[10, 20)`.
This happens when a loop over a set reads neighbor or child cells that do not exist, for example `u(1, 2 * i)` on a coarse interval whose children are not all in the mesh.

**Fix**: build the set you loop over so that every interval you read exists: intersect it with the cells of the level you read, or contract it.
The tutorial [Apply an operator on a set](../tutorial/operator_on_subset.md#projecting-on-the-intersection) reproduces both messages and fixes them with `samurai::contract`.

## A restart with another number of processes

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  The number of processes in the restart file (4) does not match the current number of processes (2).
```

**Cause**: `samurai::load` read a checkpoint written by a different number of MPI processes.
Each process reads back the part of the file written by the process of the same rank, so the counts must match.

**Fix**: run the restarted program with the number of processes given first in the message, here `mpiexec -n 4`.
See [Restart with the same number of MPI processes](restart.md#restart-with-the-same-number-of-mpi-processes) in the restart guide.

## A restarted field that is not in the checkpoint

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  The field rho does not exist in the file.
```

**Cause**: `samurai::load` finds the fields in the file by their names, and no field of the checkpoint has the name of the field you passed.

**Fix**: create the field to load with the name it had when you called `samurai::dump`.
See [Load the checkpoint](restart.md#load-the-checkpoint) in the restart guide.

## A restarted field without a name

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  The field has no name.
```

**Cause**: a field passed to `samurai::load` has an empty name, so it cannot be found in the file.

**Fix**: give the field the name it had when you called `samurai::dump`, for example `samurai::make_scalar_field<double>("u", mesh)`.
See [Load the checkpoint](restart.md#load-the-checkpoint) in the restart guide.

## A restarted field with another number of components

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  The number of components of the field (2) does not match the expected number of components (1).
```

**Cause**: the field in the checkpoint (first number) and the field you passed to `samurai::load` (second number) do not have the same number of components, for example a vector field dumped and a scalar field loaded under the same name.

**Fix**: create the field to load with the type and number of components of the dumped field.
See [Load the checkpoint](restart.md#load-the-checkpoint) in the restart guide.

## A solver without unknown

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  Undefined unknown(s) for this linear system. Please set the unknowns using the instruction '[solver].set_unknown(u);' or '[solver].set_unknowns(u1, u2...);'.
```

**Cause**: `solve` was called on a solver made by `samurai::petsc::make_solver` before the solver knew which field to solve for.

**Fix**: call `solver.set_unknown(u)` (or `set_unknowns(u1, u2, ...)` for a block system) before `solver.solve(rhs)`, or pass the unknown with `solver.solve(u, rhs)`.
`samurai::petsc::solve(D, u, rhs)` sets the unknown itself.

## A non-linear scheme without Jacobian

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  The jacobian function of operator 'burgers' has not been implemented.
Use option -snes_mf or -snes_fd for an automatic computation of the jacobian matrix.
```

**Cause**: an implicit solve assembles the Jacobian matrix of a non-linear scheme that has no Jacobian function.
The name in quotes is the name given with `set_name`.

**Fix**: define `cons_jacobian_function` next to `cons_flux_function` (or `jacobian_function` for a non-conservative scheme), or run the program with `-snes_mf`, which solves without assembling the matrix.
The message also names `-snes_fd`; if the solver then stops with `Divergence of the non-linear solver (DIVERGED_LINEAR_SOLVE)`, use `-snes_mf` instead.
See [Jacobian function](../reference/finite_volume_schemes.md#jacobian-function) in the finite volume schemes reference.

## The solver diverges

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  Divergence of the solver (DIVERGED_ITS)
```

or, for a non-linear solver:

```text
terminate called after throwing an instance of 'std::runtime_error'
  what():  Divergence of the non-linear solver (DIVERGED_LINE_SEARCH)
```

**Cause**: the PETSc linear solver (KSP) or non-linear solver (SNES) stopped without converging.
The text in parentheses is the PETSc converged reason, for example `DIVERGED_ITS` (too many iterations) or `DIVERGED_LINE_SEARCH` (the Newton step cannot reduce the residual).

**Fix**: run the program with `-ksp_converged_reason` and `-snes_converged_reason` (add `-ksp_monitor` or `-snes_monitor` to follow the iterations) to see where the solve fails.
Check the boundary conditions attached to the unknown, the time step and, for a non-linear scheme, the Jacobian function and the initial guess in the unknown field.
On a small system, try a direct solver (`-ksp_type preonly -pc_type lu`): if it fails too, the system itself is singular or badly posed.
To keep running after a non-linear solve diverges, call `stop_program_on_divergence(false)` on the non-linear solver.

## A PETSc solver destroyed after samurai::finalize

With Open MPI:

```text
*** The MPI_Comm_get_attr() function was called after MPI_FINALIZE was invoked.
*** This is disallowed by the MPI standard.
*** Your MPI job will now abort.
```

With MPICH:

```text
Attempting to use an MPI routine (MPII_Comm_get_attr) before initializing or after finalizing MPICH
```

The program aborts after its last line of output, even when it runs on one process.

**Cause**: a solver made by `samurai::petsc::make_solver` lives until the end of `main`, after `samurai::finalize()` has finalized PETSc and MPI.
Its destructor frees PETSc objects, which calls MPI.

**Fix**: destroy the solvers before `samurai::finalize()`.
Create them inside a block `{ ... }` that closes before the call, or inside a function that returns before it.
