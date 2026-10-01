# How-to: save your samurai mesh and fields

This guide shows you how to write a samurai mesh and its fields to disk so that you can post-process and visualize them.
The `save` function writes an HDF5 file and an XDMF file that ParaView opens directly.

samurai has two families of output functions:

- `save` writes the mesh and fields for post-processing: an HDF5 file and an XDMF file.
- `dump` and `load` write and read restart files: an HDF5 file only, with no XDMF file.

## Before you start

- You have a mesh and fields. If not, see the [mesh how-to guide](mesh.md) and the [field how-to guide](field.md).
- Include `samurai/io/hdf5.hpp` to use `save`, and `samurai/io/restart.hpp` to use `dump` and `load`.
- Call `samurai::initialize(argc, argv)` at the start of `main` and `samurai::finalize()` at the end if your build uses MPI or if you want the `--save-debug-fields` option.
  `initialize` starts MPI and reads the samurai command-line options.
- For an MPI build, HDF5 must be built with parallel (MPI-IO) support.

## Save a mesh

Call `save` with an output directory, a file name without extension and the mesh:

```{literalinclude} snippet/save/save_mesh.cpp
    :language: c++
```

The first call writes `output_path/mesh_filename.h5` and `output_path/mesh_filename.xdmf`.
If `output_path` does not exist, `save` creates it.
`save` creates only the last directory of the path: the parent directories must already exist.

If you leave out the directory, as in the second call, `save` writes the files in the current working directory.

```{warning}
`save` overwrites any existing file with the same name.
To keep one file per time step, add the step number to the file name, for example with `fmt::format("{}_ite_{}", filename, nsave)`.
```

## Save fields with the mesh

Pass the fields after the mesh:

```{literalinclude} snippet/save/save_field.cpp
    :language: c++
```

This writes `output_path/fields.h5` and `output_path/fields.xdmf` with the scalar field `u` and the vector field `v`.

- The mesh must come right after the file name (or after the options, see {ref}`howto-save-submeshes`).
- You can pass any number of fields.
- A vector field is written as one dataset per component: `v_0`, `v_1` and `v_2` for the field `v` above.
- `save` does not check that the fields are defined on the mesh you pass: make sure they are.

## Add debug fields

To add the fields `indices`, `coordinates` (cell centers) and `levels` to every file that `save` writes, run your program with the `--save-debug-fields` option:

```bash
./your_program --save-debug-fields
```

The option is read by `samurai::initialize(argc, argv)`. Without that call, the option has no effect.

(howto-save-submeshes)=

## Save the sub-meshes

A samurai mesh holds several sub-meshes.
A `UniformMesh` holds the cells and the cells with their ghost cells.
An `MRMesh` also holds the sub-meshes used to compute the details.
By default, `save` writes only the cells.

To write the sub-meshes as well, pass a `samurai::Hdf5Options` object after the file name:

```{literalinclude} snippet/save/save_all_submeshes.cpp
    :language: c++
```

The arguments of `Hdf5Options` depend on the mesh type:

| Mesh type | Arguments | Effect |
|---|---|---|
| `MRMesh`, `AMRMesh` | `{by_level, by_mesh_id}` | `by_level` writes one grid per level, named `Level <level>`, from `min_level - 2` (or 0) to `max_level`. `by_mesh_id` writes one grid per sub-mesh, named after its mesh id (`cells`, `cells and ghosts`, ...). With both set, each level holds one grid per sub-mesh. |
| `CellArray` | `{by_level, by_mesh_id}` | `by_level` works as for `MRMesh`. `by_mesh_id` writes the cell array as one grid named `cell_array`. |
| `UniformMesh` | `{by_mesh_id}` | `by_mesh_id` writes the `cells` and `cells and ghosts` sub-meshes as separate grids. A `UniformMesh` has one level, so there is no `by_level` argument. |
| `LevelCellArray` | `{by_level, by_mesh_id}` | Only `by_mesh_id` has an effect, and it writes the single sub-mesh as its own grid. |

Both arguments default to `false`.

When you leave out the directory, pass the options as a named object and not as a braced list:

```cpp
samurai::Hdf5Options<decltype(mesh)> options(true, true);
samurai::save("fields", options, mesh, field_1, field_2);
```

The call `samurai::save("fields", {true, true}, mesh, ...)` also matches the overload that takes a directory and a file name (`{true, true}` converts to a `std::string`), so the call is either ambiguous or resolved to the wrong overload.
With a directory, as in the snippet above, the braced list works.

## If you run with MPI

- `save` is collective over `MPI_COMM_WORLD`: every rank must call it with the same directory and file name.
- All ranks write into one HDF5 file. Rank 0 writes the XDMF file.
  In the HDF5 file, the data of each rank is stored under `/mesh/rank_<rank>/`.
- To use another communicator, pass it after the file name: `samurai::save(path, filename, comm, mesh, fields...)`.
- To write one file per rank, use `samurai::local_save(path, filename, mesh, fields...)`.
  Each rank writes `<filename>_rank<rank>.h5` and `<filename>_rank<rank>.xdmf`.

## Check the result

1. List the output directory. It contains `fields.h5` and `fields.xdmf`.
2. List the content of the HDF5 file:

   ```bash
   h5ls -r output_path/fields.h5
   ```

   For a run on one process, the file holds the datasets `/mesh/points`, `/mesh/connectivity`, `/mesh/fields/u` and `/mesh/fields/v_0` to `/mesh/fields/v_2`.
3. Open `fields.xdmf` (not the `.h5` file) in ParaView with **File > Open**. The fields `u`, `v_0`, `v_1` and `v_2` appear in the list of cell arrays.

## Save for a restart

To write a file that your program can read back, use `dump` and `load`:

```{literalinclude} snippet/save/dump_mesh.cpp
    :language: c++
```

`dump` writes `restart_file.h5` in the current directory.
`load` reads it back: it replaces the mesh with the saved one and fills the fields that have the same names.

- `dump` writes no XDMF file. Use `save` for files that you open in ParaView.
- `dump` does not create the directory: it must exist.
- The fields you pass to `load` must have the same names and number of components as the dumped fields.

```{warning}
In an MPI program, run `load` with the same number of MPI processes as `dump`.
Otherwise, `load` throws `std::runtime_error` with the message `The number of processes in the restart file (...) does not match the current number of processes (...)`.
```

## Related

- [How-to: plot samurai fields and meshes](plot.md), to visualize the files that `save` writes.
- [How-to: set options in samurai](options.md), for the other samurai command-line options.
