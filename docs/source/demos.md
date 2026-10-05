# Demos

The `demos/` folder of the {{ project }} repository holds complete programs: finite volume and lattice Boltzmann solvers, the programs of the tutorials, MPI and load balancing tests, and comparisons with other adaptive mesh libraries.
This page lists them by family, with the CMake target of each program and a command that runs it.
To see what {{ project }} simulations look like, with videos of the solution and the mesh, see the {doc}`samurai gallery <gallery>`.

## Build and run a demo

The demos are built from a clone of the repository, as in {ref}`build-from-source`.

1. Configure {{ project }} with the options the demo needs (see [Build options](#build-options)):

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON
   ```

   `BUILD_DEMOS=ON` adds every demo to the default build.
   Without it, the demos are still configured, but each one is built only when you name its target.

2. Build the target of the demo:

   ```bash
   cmake --build build --target finite-volume-advection-2d
   ```

3. Run the program from its folder in the build tree, which mirrors `demos/`:

   ```bash
   cd build/demos/FiniteVolume
   ./finite-volume-advection-2d --Tf 0.05 --nfiles 20
   ```

   In each family below, the run commands start from the folder of the family in the build tree.

Every demo except `mesh-from-obj` accepts the predefined command-line options of {{ project }}, such as `--min-level`, `--max-level` and `--mr-eps` (see {doc}`howto/options`).
Most demos also read `--Tf` (final time), `--nfiles` (number of output files), `--path` and `--filename` (where the HDF5 and XDMF output goes).
`--help` lists the options of each program.
Open the output as shown in {doc}`howto/plot`.

(build-options)=

### Build options

Most demos need only the dependencies of {{ project }}.
The others are configured only when an option is on; their target does not exist otherwise.

| CMake option | Demos it enables | Extra dependency |
| --- | --- | --- |
| `WITH_PETSC=ON` | the implicit finite volume demos, `multigrid` and `highorder` | PETSc, found with pkg-config |
| `WITH_MPI=ON` | `finite-volume-burgers-os-2d-mpi` and the `demos/mpi` family | MPI and HDF5 built with MPI |
| `SAMURAI_WITH_PARMETIS=ON` | the `metis` strategy in the load balancing demos | ParMETIS, with `WITH_MPI=ON` |
| `SAMURAI_WITH_PTSCOTCH=ON` | the `scotch` strategy in the load balancing demos | PT-Scotch, with `WITH_MPI=ON` |
| `WITH_CGAL=ON` | `mesh-from-obj` | CGAL and Eigen 3.1 or later |

## Tutorial programs

Folder `demos/tutorial`.
These programs go with the {doc}`tutorials <tutorial/index>`.

| Target | What it does | Explained in |
| --- | --- | --- |
| `tutorial-getting-started` | Advects a disc on a 2D multiresolution mesh. | {doc}`tutorial/getting_started` |
| `tutorial-interval` | Builds a 2D `CellArray` from a list of intervals and from a box, and prints both. | {doc}`tutorial/interval` |
| `tutorial-2d-mesh` | Builds a 2D mesh from a `CellList` and saves it. | {doc}`tutorial/interval` |
| `tutorial-set-operator` | Finds where two levels of a 1D mesh overlap with the set algebra, and computes the field projection on that set. | {doc}`tutorial/operator_on_subset` |
| `tutorial-graduation-case-1` | Refines a 2D mesh of cells at several levels, without overlap, until it is graded. | {doc}`tutorial/graduation_case_1` |
| `tutorial-graduation-case-2` | Removes the overlaps of a 2D mesh with cells at several levels, then grades it. | {doc}`tutorial/graduation_case_2` |
| `tutorial-graduation-case-3` | Refines a graded mesh with a criterion and keeps it graded (`--with-graduation`). | {doc}`tutorial/graduation_case_3` |
| `tutorial-reconstruction-1d`, `-2d`, `-3d` | Adapts a function with multiresolution, rebuilds it on the finest level and writes the error. `--case` is `abs`, `exp` or `tanh`. | |
| `tutorial-proj-on-mesh` | Transfers a 3D field from one adapted mesh to another with `samurai::transfer`. | |
| `tutorial-use-config-file` | Reads `--min-level` and `--max-level` from a TOML file and checks their values. | {ref}`howto-options-config` |

Run commands, from `build/demos/tutorial`:

```bash
./tutorial-getting-started
./tutorial-interval
./tutorial-2d-mesh
./tutorial-set-operator
./tutorial-graduation-case-1
./tutorial-graduation-case-2
./tutorial-graduation-case-3 --with-graduation
./tutorial-reconstruction-2d --case tanh --mr-reg 10
./tutorial-proj-on-mesh
./tutorial-use-config-file --config ../../../demos/tutorial/test_config.toml
```

### 1D Burgers, step by step

Folder `demos/tutorial/AMR_1D_Burgers`, one subfolder per step.
Each step adds one piece of an AMR solver for the 1D Burgers equation, written by hand with intervals: the mesh, the field, the refinement criterion, the graduation, the mesh update and the ghost update.
The {doc}`algorithm tutorial <tutorial/algorithm>` reads steps 1 and 2.
The {doc}`1D Burgers tutorial <tutorial/1d_burgers_amr>` solves the same problem with the functions of {{ project }} that replace this code.

The targets are `tutorial-burgers1d-step-0` to `tutorial-burgers1d-step-6`.
Run step `<n>` from `build/demos/tutorial/AMR_1D_Burgers/step_<n>`:

```bash
./tutorial-burgers1d-step-<n>
```

## Finite volume

Folder `demos/FiniteVolume`.
These solvers use the finite volume schemes of {{ project }} (see {doc}`reference/finite_volume_schemes`) on multiresolution meshes, except `finite-volume-amr-burgers-hat`, which uses AMR.

### Explicit schemes

These demos need no extra CMake option.

| Target | Problem |
| --- | --- |
| `finite-volume-advection-1d` | 1D advection, upwind scheme, with a restart file. |
| `finite-volume-advection-2d` | 2D advection of a disc, with a restart file and load balancing under MPI. See {doc}`howto/adapt` and {doc}`howto/restart`. |
| `finite-volume-advection-2d-user-bc` | 2D advection with a boundary condition written by the user. See {ref}`reference-bc-custom`. |
| `finite-volume-advection-3d` | 3D advection of a sphere. |
| `finite-volume-scalar-burgers-2d` | 2D scalar Burgers equation. |
| `finite-volume-burgers` | 2D Burgers equation with a WENO5 scheme; `--init-sol` is `hat`, `linear` or `bands`. |
| `finite-volume-burgers-mra` | 1D Burgers equation from a hat or a Gaussian, on a multiresolution mesh. |
| `finite-volume-burgers-os` | 1D Burgers equation with a one-step high-order scheme (`convection_nonlin_os.hpp`). |
| `finite-volume-amr-burgers-hat` | 1D Burgers equation from a hat, on an AMR mesh graded with `samurai::graduation`. See {doc}`philosophy`. |
| `finite-volume-level-set-amr` | 2D level-set transport in a vortex, on an AMR mesh. See {doc}`tutorial/level_set`. |
| `finite-volume-level-set-mra` | The same problem on a multiresolution mesh. See {doc}`tutorial/level_set`. |
| `finite-volume-level-set-from-scratch` | The same problem with the schemes written in the program. |
| `finite-volume-linear-convection-obstacle` | 2D convection in a domain with a hole. See {doc}`howto/geometry`. |

Run commands, from `build/demos/FiniteVolume`:

```bash
./finite-volume-advection-1d --Tf 0.1
./finite-volume-advection-2d --Tf 0.05 --nfiles 20
./finite-volume-advection-2d-user-bc --Tf 0.05
./finite-volume-advection-3d --Tf 0.05
./finite-volume-scalar-burgers-2d --Tf 0.001
./finite-volume-burgers --init-sol hat --min-level 3 --max-level 6 --Tf 0.1 --nfiles 1
./finite-volume-burgers-mra --init-sol hat --min-level 2 --max-level 9 --mr-eps 1e-5 --nfiles 1
./finite-volume-burgers-os --Tf 0.6 --min-level 3 --max-level 10 --mr-eps 1e-3 --nfiles 1
./finite-volume-amr-burgers-hat --Tf 1
./finite-volume-level-set-amr --Tf 0.1
./finite-volume-level-set-mra --Tf 0.1
./finite-volume-level-set-from-scratch --Tf 0.1
./finite-volume-linear-convection-obstacle --Tf 0.3 --nfiles 1
```

### Implicit schemes (PETSc)

These demos need `WITH_PETSC=ON`.
Options that start with a single dash, such as `-pc_type lu`, go to the PETSc solver.

| Target | Problem |
| --- | --- |
| `finite-volume-heat` | 2D heat equation, implicit or explicit (`--explicit`). |
| `finite-volume-heat-heterogeneous` | 2D heat equation with a diffusion coefficient that depends on the cell. See {ref}`lin_heter_operators`. |
| `finite-volume-heat-nonlinear` | 2D nonlinear heat equation, with a flux written by the user. Its printed L2 error is wrong ([#610](https://github.com/hpc-maths/samurai/issues/610)). |
| `finite-volume-nagumo` | 1D Nagumo reaction-diffusion equation, implicit or IMEX. See {doc}`tutorial/reaction_diffusion`. |
| `finite-volume-linear-convection` | 2D periodic convection with WENO5, explicit or backward Euler (`--implicit`). |
| `finite-volume-stokes-2d` | 2D Stokes problem, stationary (`s`) or not (`ns`), solved as a block system. |
| `finite-volume-lid-driven-cavity` | Lid-driven cavity, incompressible Navier-Stokes. |
| `finite-volume-diff-heated-cavity` | Differentially heated cavity, Navier-Stokes with a Boussinesq buoyancy term. |
| `manual_block_matrix_assembly` | Assembles a block matrix with hand-written PETSc blocks (`samurai::petsc::ManualAssembly`). |

Run commands, from `build/demos/FiniteVolume`:

```bash
./finite-volume-heat --init-sol crenel --Tf 0.1 --min-level 3 --max-level 8 -pc_type lu
./finite-volume-heat-heterogeneous --Tf 0.1 --min-level 3 --max-level 6 -ksp_type preonly -pc_type lu
./finite-volume-heat-nonlinear --Tf 0.1 --min-level 3 --max-level 6
./finite-volume-nagumo --min-level 4 --max-level 8 --Tf 0.1 --dt 0.02
./finite-volume-linear-convection --min-level 1 --max-level 6 --Tf 0.1 --nfiles 1
./finite-volume-stokes-2d --test-case s --min-level 5 --max-level 5
./finite-volume-lid-driven-cavity --min-level 3 --max-level 6 --Tf 0.03 --nfiles 1 -pc_type lu
./finite-volume-diff-heated-cavity --Tf 0.7 --nfiles 1
./manual_block_matrix_assembly
```

### MPI

This demo needs `WITH_MPI=ON`.

| Target | Problem |
| --- | --- |
| `finite-volume-burgers-os-2d-mpi` | 2D Burgers equation with a one-step high-order scheme, on a grid of `--npx` by `--npy` processes. |

Run commands, from `build/demos/FiniteVolume`:

```bash
mpiexec -n 4 ./finite-volume-burgers-os-2d-mpi --npx=2 --npy=2 --min-level=0 --max-level=8 --Tf=6 --nfiles 1
```

`finite-volume-advection-2d`, `finite-volume-burgers` and `finite-volume-heat` also run with `mpiexec` when {{ project }} is built with `WITH_MPI=ON`, and the CI compares their parallel and sequential results.
With `--load-balancing-at N`, they rebalance the mesh along a Hilbert curve every `N` mesh adaptations:

```bash
mpiexec -n 4 ./finite-volume-advection-2d --Tf 0.05 --nfiles 20 --load-balancing-at 10
```

## Lattice Boltzmann

Folder `demos/LBM`.
These programs use the `schemes/lbm` header and need no extra CMake option.
{doc}`LBM/test_cases` describes each problem, its parameters and the papers it comes from.

| Target | Problem |
| --- | --- |
| `lbm-new-D1Q2-advection` | D1Q2, 1D advection or Burgers (`--burgers`). |
| `lbm-new-D2Q4-advection` | D2Q4, 2D advection. |
| `lbm-new-D2Q4diag-advection` | D2Q4 with diagonal velocities, 2D advection. |
| `lbm-new-D1Q3-shallow-waters-dam` | D1Q3, shallow water dam break with wall boundaries. |
| `lbm-new-D1Q5-shallow-waters-dam` | D1Q5, shallow water dam break. |
| `lbm-new-D1Q222-euler-sod` | D1Q222, Euler equations, Sod shock tube. |
| `lbm-new-D2Q4444-euler-lax-liu` | D2Q4444, Euler equations, Lax-Liu Riemann problems (`--riemann`). |
| `lbm-new-D2Q4444-euler-implosion` | D2Q4444, Euler equations, implosion in a closed box. |
| `lbm-new-D2Q5444-euler-rayleigh-taylor` | D2Q5444, Euler equations with gravity, Rayleigh-Taylor instability. |
| `lbm-new-D2Q9-taylor-green` | D2Q9 MRT, Taylor-Green vortex. |
| `lbm-new-D2Q9-von-karman` | D2Q9 MRT, flow past a cylinder (von Kármán street). |

Run commands, from `build/demos/LBM`:

```bash
./lbm-new-D1Q2-advection --adapt --level 8 --min-lvl 3 --eps 1e-4 --Tf 0.2
./lbm-new-D2Q4-advection --level 5 --Tf 0.2
./lbm-new-D2Q4diag-advection --adapt --level 7 --min-lvl 2 --eps 1e-4 --Tf 0.2
./lbm-new-D1Q3-shallow-waters-dam --bc bounceback --level 7 --hL 2 --hR 1 --Tf 0.5
./lbm-new-D1Q5-shallow-waters-dam --min-level 3 --max-level 8 --mr-eps 1e-4 --Tf 0.2 --nfiles 1
./lbm-new-D1Q222-euler-sod --min-level 3 --max-level 9 --mr-eps 1e-3 --Tf 0.2 --nfiles 1
./lbm-new-D2Q4444-euler-lax-liu --min-level 2 --max-level 6 --mr-eps 1e-3 --Tf 0.1 --nfiles 1
./lbm-new-D2Q4444-euler-implosion --min-level 2 --max-level 6 --mr-eps 1e-3 --Tf 0.3 --nfiles 1
./lbm-new-D2Q5444-euler-rayleigh-taylor --min-level 2 --max-level 7 --mr-eps 1e-3 --Tf 0.5 --nfiles 1
./lbm-new-D2Q9-taylor-green --level 5 --U0 0.05 --nu 0.02 --Tf 2
./lbm-new-D2Q9-von-karman --min-level 4 --max-level 6 --mr-eps 1e-3 --Tf 1 --nfiles 1
```

## MPI and load balancing

Folder `demos/mpi`.
These programs need `WITH_MPI=ON`, and are not configured otherwise.
{ref}`reference-load-balancing-tools` describes the options of the load balancing demos.

| Target | What it does |
| --- | --- |
| `mpi-load-balancing-2d` | Advects a disc on a periodic adaptive mesh and rebalances it with the strategy given by `--lb-strategy`. |
| `mpi-load-balancing-3d` | The same problem in 3D, with a sphere. |
| `mpi-load-balancing-tube` | Balances a 2D mesh refined around circles in a tube of length `--length`. |
| `mpi-ghost-cases` | Writes the meshes and the decompositions checked by the parallel ghost update test. |

Run commands, from `build/demos/mpi`:

```bash
mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy sfc-hilbert --Tf 0.02 --nt-loadbalance 5
mpiexec -n 4 ./mpi-load-balancing-3d --lb-strategy diffusion --Tf 0.06 --nt-loadbalance 2
mpiexec -n 4 ./mpi-load-balancing-tube --lb-strategy sfc-hilbert
mpiexec -n 4 ./mpi-ghost-cases --path ghost_cases
```

## Solvers and multigrid

Folders `demos/multigrid` and `demos/highorder`.
Both need `WITH_PETSC=ON`.

| Target | What it does |
| --- | --- |
| `multigrid` | Solves a Poisson problem on $[0, 1]^d$ on a uniform mesh at level `--level` with PETSc, and prints the $L^2$ error; `--tc` is `poly` or `exp`, and `-pc_type mg` selects the geometric multigrid of {{ project }}. |
| `highorder` | Solves a 2D Poisson problem with a high-order boundary condition on refined meshes, and prints the $L^2$ error and the convergence order. |

Run commands, from `build/demos/multigrid` and `build/demos/highorder`:

```bash
./multigrid --level 5
./highorder
```

```{warning}
Three of these commands fail on the current code:

- `./highorder` aborts at start ([#607](https://github.com/hpc-maths/samurai/issues/607));
- `./manual_block_matrix_assembly` aborts at exit ([#608](https://github.com/hpc-maths/samurai/issues/608));
- `./multigrid --level 5 -pc_type mg` does not converge to the right solution ([#609](https://github.com/hpc-maths/samurai/issues/609)).
```

## Mesh from a geometry

Folder `demos/from_obj`, built with `WITH_CGAL=ON`.

| Target | What it does |
| --- | --- |
| `mesh-from-obj` | Builds a graded 3D adaptive mesh around the surface of an OBJ file with `samurai::from_geometry`, and saves it. `--keep-inside` and `--keep-outside` keep the cells inside or outside the object. |

Run commands, from `build/demos/from_obj`:

```bash
./mesh-from-obj --input ../../../demos/from_obj/apple.obj --max-level 7
```

{doc}`howto/geometry` shows how the demo is built.

## Comparisons with other AMR libraries

Folders `demos/p4est` and `demos/pablo`.
These programs rebuild an example of another adaptive mesh library with {{ project }}.

| Target | What it does |
| --- | --- |
| `p4est-simple-2d` | Rebuilds the mesh of the p4est [`simple2` example](https://github.com/cburstedde/p4est/blob/master/example/simple/simple2.c) and prints the memory used by the mesh. |
| `pablo-bubble-2d` | Moves bubbles on an AMR mesh, as the [PABLO bubbles example](https://github.com/optimad/bitpit/blob/master/examples/PABLO_bubbles_2D.cpp) of bitpit. |

Run commands, from `build/demos/p4est` and `build/demos/pablo`:

```bash
./p4est-simple-2d
./pablo-bubble-2d --Tf 3
```

{doc}`philosophy` discusses the p4est comparison.

## Programs that are not built

Some files in `demos/` are kept for reference and have no CMake target:

- `demos/Weno`: `demos/CMakeLists.txt` does not add this folder.
- `demos/FiniteVolume/BZ`: the Belousov-Zhabotinsky programs; `demos/FiniteVolume/CMakeLists.txt` does not add this folder.
- `demos/LBM/test_*.cpp` and `demos/LBM/D*Q*_*.cpp`: the lattice Boltzmann programs that produced the figures of the papers, written with an older API (see {doc}`LBM/test_cases`).
