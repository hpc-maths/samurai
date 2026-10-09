# How-to: run a samurai program with MPI

This guide shows you how to build a samurai program with MPI, run it on several processes, and check that it computes the same result as on one process.
The example adapts a mesh, prints the number of cells of each process and two global values, and saves the mesh with its field.

## Before you start

- You know how to build a samurai program with CMake. If not, see the [CMake project how-to guide](cmake.md).
- The program calls `samurai::initialize(argc, argv)` at the start of `main` and `samurai::finalize()` at the end.
  In an MPI build, `initialize` calls `MPI_Init` and `finalize` calls `MPI_Finalize`.
  With PETSc, `PetscInitialize` and `PetscFinalize` do it instead.
- The mesh is built from a `samurai::Box`.
  A mesh built from a `samurai::DomainBuilder` throws `std::runtime_error` with the message `MPI is not implemented with DomainBuilder.` in an MPI build.

## Install the parallel dependencies

An MPI build of samurai needs an MPI implementation, Boost.MPI with Boost.Serialization, and an HDF5 library built with MPI support.

- With conda, install `mpich`, `libboost-mpi`, `libboost-devel`, `libboost-headers` and `'hdf5=*=mpi_*'`.
- For a build from source, create the environment of the repository, which also holds ParMETIS:

  ```bash
  conda env create --file conda/mpi-environment.yml
  conda activate samurai-mpi-env
  ```

- With spack, install the `+mpi` variant.

The [installation guide](installation.md) gives the full commands for each route.

## Build with MPI

How you turn on MPI depends on where you build:

- In your own CMake project, set `SAMURAI_WITH_MPI` before `find_package(samurai)`, or pass it on the command line:

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSAMURAI_WITH_MPI=ON
  cmake --build build
  ```

- In the samurai source tree, for the demos, tests and documentation snippets, pass `WITH_MPI`:

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=ON
  ```

  Add `-DSAMURAI_WITH_PARMETIS=ON` for the ParMETIS load balancing strategy.

Both options add the compile definition `SAMURAI_WITH_MPI` and link Boost.MPI and Boost.Serialization.
The configuration stops with `HDF5 is not parallel. Please install a parallel version.` when the HDF5 that CMake finds has no MPI support.
To check the build, run your program with `--info`: the `Build configuration` section shows `MPI : ON`.

Code that calls MPI or Boost.MPI directly must sit inside `#ifdef SAMURAI_WITH_MPI`, so that the same source still builds without MPI.
The example below does this for its reductions.

## Write the program

The program adapts a mesh to a peak near the bottom left corner of the unit square.
Each process then counts its own cells and computes the integral of `u` on its own cells.
Rank 0 gathers the cell counts with `boost::mpi::gather`, and `boost::mpi::all_reduce` sums the counts and the integrals over all processes.

```{literalinclude} snippet/mpi/mpi_adapt.cpp
  :language: c++
```

Everything else needs no MPI code: `samurai::mra::make_mesh` splits the mesh between the processes, the adaptation exchanges the cells it needs with the neighboring processes, and `samurai::save` writes all the processes into one file.
Every process must call the collective functions (the mesh constructor, the adaptation, `samurai::update_ghost_mr`, `samurai::save`) in the same order.

## Run on several processes

Start the program with your MPI launcher and the number of processes:

```bash
mpiexec -n 4 ./mpi_adapt
```

On 1, 2 and 4 processes, the program prints:

```{literalinclude} snippet/mpi/mpi_adapt_np1_output.txt
  :language: text
```

```{literalinclude} snippet/mpi/mpi_adapt_np2_output.txt
  :language: text
```

```{literalinclude} snippet/mpi/mpi_adapt_np4_output.txt
  :language: text
```

The total number of cells does not depend on the number of processes.
The integral differs in the last digit on 4 processes: each process sums its own cells first, so the additions happen in another order.

The cells are not shared evenly.
The mesh constructor splits the domain at its start level into contiguous blocks of rows, one block per process, before any adaptation.
The adaptation then refines near the peak, in the rows of ranks 0 and 1, and coarsens the rest.
Fig. 1 of {ref}`plate-mpi-subdomains` shows the four bands.

```{plate} The subdomains of four processes
:figure: mpi_subdomains
:label: plate-mpi-subdomains

**Fig. 1.** The ranks after `mpiexec -n 4 ./mpi_adapt`.
The mesh constructor gives each rank 32 rows of level 7, a quarter of the square.
The adaptation keeps every cell on its rank, so rank 1, which holds the peak, ends with most of the cells and rank 3 with 16.

**Fig. 2.** The ranks of the same program run with `--load-balancing-at 1`.
The load balancer sorts the cells along the Hilbert curve, as the {doc}`load balancing reference <../reference/load_balancing>` explains, and cuts that order into four pieces of 395, 395, 395 and 394 cells.
Rank 1 gets a small region around the peak, where the cells are finest.

*Each rank has its own tone and pattern, and heavy rules separate the ranks.
The cells of levels 6 and 7 have a fainter outline, so that their rank stays visible.
Both figures are drawn from the file `mpi_adapt_size_4.h5` of each run.*
```

### See the output of every process

`samurai::initialize` sends the standard output of every rank other than 0 to `/dev/null`, so that a run on 100 processes does not print every line 100 times.
The error output is not redirected.
Pass `--dont-redirect-output` to keep the standard output of every rank, for example while you debug:

```bash
mpiexec -n 4 ./mpi_adapt --dont-redirect-output
```

The last line of the program now comes from every process.
The processes write at the same time, so the order of these lines changes from one run to the next:

```text
rank 0: 490 cells
rank 1: 955 cells
rank 2: 118 cells
rank 3: 16 cells
total: 1579 cells
integral of u: 1.570796323809258e-02
rank 0 saved 490 cells
rank 1 saved 955 cells
rank 2 saved 118 cells
rank 3 saved 16 cells
```

The option is defined only in an MPI build.

## Balance the load

To redistribute the cells after the adaptation, pass `--load-balancing-at N`.
The adaptation built by `samurai::make_MRAdapt` then rebalances the mesh every `N` calls, starting with the first one, and moves the fields it adapts with the cells.
The default is 0, which never rebalances.

```bash
mpiexec -n 4 ./mpi_adapt --load-balancing-at 1
```

```{literalinclude} snippet/mpi/mpi_adapt_load_balancing_output.txt
  :language: text
```

Every rank now holds a quarter of the cells, and each subdomain is a piece of the Hilbert curve, as Fig. 2 of {ref}`plate-mpi-subdomains` shows.

Each rebalance moves cells and field values between processes. In a time loop, pick `N` large enough that this cost stays small, and small enough that the load does not drift far between two rebalances.
The option always uses the Hilbert space-filling curve with the same weight for every cell.
For another strategy, a weight per level, or statistics on the migration, call a `LoadBalancer` yourself, as the {doc}`load balancing reference <../reference/load_balancing>` explains.

## Save the results

`samurai::save` is collective: every process calls it with the same directory and file name, and all the processes write into one HDF5 file.
The example puts the number of processes in the file name, so that runs on different numbers of processes do not overwrite each other.

On several processes, the cells of rank `r` are stored under `/mesh/rank_<r>/` in the HDF5 file.
`h5ls -r mpi_adapt_size_4.h5` lists:

```text
/mesh/rank_0/connectivity Dataset {490, 4}
/mesh/rank_0/fields/u    Dataset {490}
/mesh/rank_0/points      Dataset {583, 3}
/mesh/rank_1/connectivity Dataset {955, 4}
...
```

On one process, the datasets sit directly under `/mesh`, as in a build without MPI.
Rank 0 writes the XDMF file alone.
It holds one spatial collection with one grid per process that owns cells, named `mesh_rank_<r>`, so ParaView opens the whole mesh at once and can color it by process.
The [save how-to guide](save.md) describes the other arguments of `save`, and `samurai::local_save`, which writes one file per process.

## Check the result against one process

On any number of processes, the mesh must have the same cells and the fields the same values as on one process, up to round-off.
`python/compare.py` checks this: it matches the cells of two files by their coordinates, whatever process owns them, and compares every field.
Give it the two file names without the `.h5` extension:

```bash
mpiexec -n 1 ./mpi_adapt
mpiexec -n 4 ./mpi_adapt
python python/compare.py mpi_adapt_size_1 mpi_adapt_size_4
```

```text
files mpi_adapt_size_1.h5 and mpi_adapt_size_4.h5 are the same
```

The script needs `h5py` and `numpy`.
It compares the fields with an absolute tolerance of `1e-12` by default; set it with `--tol`, and add a relative tolerance with `--rtol`.
When the files differ, it prints the number of cells that differ and the largest difference, and exits with a nonzero status, so you can use it in a test.
The test `tests/test_load_balancing.py` uses it this way to check every load balancing strategy against a run on one process.

## What depends on the number of processes

- The mesh and the fields match the run on one process up to round-off.
  Global sums, such as the integral above, can differ in the last digits, because the order of the additions depends on how the cells are split.
- The cells each process owns depend on the number of processes and on load balancing.
  Do not rely on a cell being on a given rank.
- The layout of the output files depends on the number of processes, but `compare.py` and ParaView read every layout.
- A checkpoint written by `samurai::dump` must be read by `samurai::load` on the same number of processes, otherwise `load` throws `std::runtime_error`.
  See the [restart how-to guide](restart.md).

## Related

- [How-to: install samurai](installation.md), for the MPI packages of each installation route.
- [How-to: create a CMake project for samurai](cmake.md), for `SAMURAI_WITH_MPI` and the other options of a downstream project.
- [How-to: adapt a mesh with multiresolution](adapt.md), for the adaptation the example runs.
- [How-to: save your samurai mesh and fields](save.md), for the output files.
- [How-to: restart a simulation from a checkpoint](restart.md), for checkpoints on several processes.
- The {doc}`load balancing reference <../reference/load_balancing>`, for the strategies, weights and options of the load balancer.
