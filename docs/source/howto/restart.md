# How-to: restart a simulation from a checkpoint

This guide shows you how to write a checkpoint of a running simulation and start a new program from it.
You write the mesh, the fields and your own metadata (time, iteration number, ...) with `samurai::dump`, then read them back into an empty mesh with `samurai::load`.

## Before you start

- You have a mesh built with `samurai::mra::make_mesh` or `samurai::amr::make_mesh`, and fields on it. If not, see the [mesh how-to guide](mesh.md) and the [field how-to guide](field.md).
- Include `samurai/io/restart.hpp`.
- Every field you want to restart has a name: `load` finds the fields in the file by their names.
- For an MPI build, HDF5 must be built with parallel (MPI-IO) support.

The example below plays both roles in one program, so that it can compare the loaded data with the original.
The first part computes a solution on an adapted 2D mesh and writes a checkpoint.
The second part is what a restarted program does: it creates an empty mesh and empty fields, and fills them from the checkpoint.

```{literalinclude} snippet/restart/restart_mesh_fields.cpp
  :language: c++
```

## Share the mesh configuration

The restarted program must build its mesh from the same configuration as the program that wrote the checkpoint.
`load` keeps the configuration of the mesh you pass to it (stencil radius, graduation width, periodicity, ...) and takes only the minimum and maximum levels from the file.
The example puts the configuration in one function used by both parts:

```{literalinclude} snippet/restart/restart_mesh_fields.cpp
  :language: c++
  :start-at: auto make_config()
  :end-at: "}"
```

## Write a checkpoint

Call `dump` with a directory, a file name without extension, a callback that writes the metadata, the mesh and the fields:

```{literalinclude} snippet/restart/restart_mesh_fields.cpp
  :language: c++
  :start-after: // Write the checkpoint
  :end-before: // End of the checkpoint
  :dedent:
```

This writes `checkpoints/checkpoint.h5`, and no XDMF file: for post-processing, use `samurai::save` as the [save how-to guide](save.md) explains.
`dump` does not create the directory: create it first, as the example does with `std::filesystem::create_directories`.
`dump` overwrites any existing file with the same name; to keep several checkpoints, add the iteration number to the file name.

The checkpoint holds:

- the number of MPI processes that wrote it;
- the real cells of the mesh, level by level, with its origin point, scaling factor, level range and whether its domain is a box;
- the values of each field on the real cells, with its name and number of components;
- the metadata your callback writes.

The callback receives a `samurai::MetadataWriter`.
Store the simulation time with `time(t)`, and any other value with `attribute(name, value)`: a scalar, a `std::string` or a `std::vector`.
Both calls return the writer, so you can chain them.
The values are stored as attributes of the root group of the HDF5 file.

The callback is optional: `samurai::dump(path, "checkpoint", mesh, u, v)` writes the same file without metadata.
Without the directory, `samurai::dump("checkpoint", mesh, u, v)` writes in the current working directory.

## Restart from the checkpoint

### Create an empty mesh and its fields

Create the mesh with `samurai::mra::make_empty_mesh` (or `samurai::amr::make_empty_mesh` for an AMR mesh), from the configuration of the first run.
Create the fields to restore on it, with the same names, value types and numbers of components as the dumped fields:

```{literalinclude} snippet/restart/restart_mesh_fields.cpp
  :language: c++
  :start-at: auto new_mesh
  :end-at: auto new_v
  :dedent:
```

The empty mesh has no cells, but it holds the configuration.

### Load the checkpoint

Call `load` with the same directory and file name, a callback that reads the metadata, the empty mesh and the fields:

```{literalinclude} snippet/restart/restart_mesh_fields.cpp
  :language: c++
  :start-at: double new_t
  :end-before: // The checkpoint holds no
  :dedent:
```

`load` replaces the cells of the mesh with the dumped ones, then resizes each field and fills it with the values stored under its name.
The callback runs after the mesh and the fields are loaded.
It receives a `samurai::MetadataReader`: read the time with `time()` and the other values with `attribute<T>(name)`, where `T` is the type you wrote.
To test whether a value exists, use `has(name)`.

Without metadata, call `samurai::load(path, "checkpoint", mesh, u, v)`, or `samurai::load("checkpoint", mesh, u, v)` for a file in the current working directory.

`load` throws `std::runtime_error` when:

- a field has no name, or no field with its name is in the file;
- the number of components of a field differs from the dumped one;
- the number of MPI processes differs from the one that wrote the file (see below).

### Restore what the checkpoint does not hold

The checkpoint holds only the values on the real cells. After `load`:

- Attach the boundary conditions of each field again with `samurai::make_bc`.
- Create the other fields of the simulation (for example the field of the next time step) after `load`, so that they have the size of the loaded mesh. A field created before `load` and not passed to it keeps its old size: call `resize()` on it.
- Ghosts are not stored. Before a loop of your own that reads the neighbors of a cell, call `samurai::update_ghost_mr` as the [adaptation how-to guide](adapt.md) explains.

The demo `demos/FiniteVolume/advection_2d.cpp` follows this pattern with a `--restart-file` option: it creates an empty mesh, loads the file when the option is given, and builds the mesh from a box otherwise.

## Restart with the same number of MPI processes

Run the restarted program with the same number of MPI processes as the program that wrote the checkpoint.
Each process writes its own part of every dataset, and on restart each process reads back the part written by the process of the same rank, so it gets back its own subdomain.
`load` compares the number of processes stored in the file with the current one and throws `std::runtime_error` if they differ.
For a checkpoint written by 4 processes and loaded by 2, the message is `The number of processes in the restart file (4) does not match the current number of processes (2).`

`dump` and `load` are collective: every process must call them with the same directory and file name.
The metadata callbacks run on every process too, so write and read the same values everywhere.

## Check the result

Count the cells of each level in the dumped and in the loaded mesh, compare the field values cell by cell, and print the metadata.
The end of the example does this. Run it:

```bash
./restart_mesh_fields
```

```text
Dumped mesh:
  level 3: 0 cells
  level 4: 160 cells
  level 5: 212 cells
  level 6: 476 cells
  level 7: 848 cells
  total: 1696 cells
Loaded mesh:
  level 3: 0 cells
  level 4: 160 cells
  level 5: 212 cells
  level 6: 476 cells
  level 7: 848 cells
  total: 1696 cells
Same mesh: true
Max difference of u and v: 0
Time: 0.25, iteration: 10
```

The loaded mesh has the same cells as the dumped one on every level, the field values are identical, and the time and iteration number come back from the metadata.

To look inside a checkpoint without a program, list its content:

```bash
h5ls -r checkpoints/checkpoint.h5
```

The file holds the `/n_process` dataset, the `/mesh` group with one group per level that has cells, and one group per field under `/fields` (`/fields/u` and `/fields/v` here).

## Related

- [How-to: save your samurai mesh and fields](save.md), to write files for post-processing.
- [How-to: plot samurai fields and meshes](plot.md), which explains how to open a checkpoint in ParaView or yt.
- [How-to: adapt a mesh with multiresolution](adapt.md), for the adaptation that follows a restart in the time loop.
