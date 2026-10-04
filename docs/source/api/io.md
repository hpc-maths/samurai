# Input and output

This page documents the HDF5 input and output of {{ project }}.
`samurai::save` writes a mesh and its fields to an HDF5 file and an XDMF file for visualization; `samurai::Hdf5Options` selects the sub-meshes it writes.
`samurai::dump` writes a checkpoint, an HDF5 file only, that `samurai::load` reads back into a mesh and fields.
`samurai::MetadataWriter` and `samurai::MetadataReader` give the callbacks of these functions access to user metadata, such as the simulation time.
Include `samurai/io/hdf5.hpp` for `save` and `samurai/io/restart.hpp` for `dump` and `load`.
For examples, see the {doc}`save how-to <../howto/save>` and the {doc}`restart how-to <../howto/restart>`.

## Save for visualization

`save` takes an optional directory, a file name without extension, optional `Hdf5Options`, the mesh and any number of fields.
It writes `<filename>.h5` and `<filename>.xdmf`, and creates the directory if it does not exist.
Without a directory, it writes in the current working directory.

```{doxygenfunction} samurai::save(const fs::path &path, const std::string &filename, const Hdf5Options<mesh_t> &options, const mesh_t &mesh, const T&... fields)
```

```{doxygenfunction} samurai::save(const fs::path &path, const std::string &filename, const mesh_t &mesh, const T&... fields)
```

```{doxygenfunction} samurai::save(const std::string &filename, const Hdf5Options<mesh_t> &options, const mesh_t &mesh, const T&... fields)
```

```{doxygenfunction} samurai::save(const std::string &filename, const mesh_t &mesh, const T&... fields)
```

With MPI, `save` is collective over `MPI_COMM_WORLD`.
The MPI build adds four overloads that take an `MPI_Comm` after the file name, such as `save(path, filename, comm, options, mesh, fields...)`; this reference is generated without MPI and does not render them.
`samurai::local_save` takes the same arguments as `save` and writes one pair of files per rank.

```{doxygenfunction} samurai::local_save(const fs::path &path, const std::string &filename, const Hdf5Options<mesh_t> &options, const mesh_t &mesh, const T&... fields)
```

```{doxygenfunction} samurai::local_save(const fs::path &path, const std::string &filename, const mesh_t &mesh, const T&... fields)
```

```{doxygenfunction} samurai::local_save(const std::string &filename, const Hdf5Options<mesh_t> &options, const mesh_t &mesh, const T&... fields)
```

```{doxygenfunction} samurai::local_save(const std::string &filename, const mesh_t &mesh, const T&... fields)
```

## Output options

`Hdf5Options<Mesh>` takes `by_level` and `by_mesh_id`, both `false` by default: `by_level` writes one grid per level and `by_mesh_id` one grid per mesh id.
The specialization `Hdf5Options<UniformMesh<Config>>`, for a mesh with a single level, has no `by_level`: its constructor takes only `by_mesh_id`.
The `metadata` member is a callback that `save` calls with a `samurai::MetadataWriter`.

```{doxygenstruct} samurai::Hdf5Options
:members:
:undoc-members:
```

## Checkpoint and restart

`dump` writes the mesh and the fields to `<filename>.h5`, overwriting any existing file; it does not create the directory.
`load` reads such a file into a mesh and fields: it rebuilds the mesh from the cells in the file and finds the fields by their names.
For a multiresolution or AMR mesh, the new mesh takes its minimum and maximum levels from the file and the rest of its configuration from the mesh passed to `load`.
It throws `std::runtime_error` when the number of MPI processes differs from the one that wrote the file.
The overloads with a `metadata` callback call it with a `samurai::MetadataWriter` after `dump` writes the file, and with a `samurai::MetadataReader` after `load` reads it.

```{doxygenfunction} samurai::dump(const fs::path &path, const std::string &filename, const Mesh &mesh, const Fields&... fields)
```

```{doxygenfunction} samurai::dump(const fs::path &path, const std::string &filename, Callback &&metadata, const Mesh &mesh, const Fields&... fields)
```

```{doxygenfunction} samurai::dump(const std::string &filename, const Mesh &mesh, const Fields&... fields)
```

```{doxygenfunction} samurai::dump(const std::string &filename, Callback &&metadata, const Mesh &mesh, const Fields&... fields)
```

```{doxygenfunction} samurai::load(const fs::path &path, const std::string &filename, Mesh &mesh, Fields&... fields)
```

```{doxygenfunction} samurai::load(const fs::path &path, const std::string &filename, Callback &&metadata, Mesh &mesh, Fields&... fields)
```

```{doxygenfunction} samurai::load(const std::string &filename, Mesh &mesh, Fields&... fields)
```

```{doxygenfunction} samurai::load(const std::string &filename, Callback &&metadata, Mesh &mesh, Fields&... fields)
```

## Metadata

```{doxygenclass} samurai::MetadataWriter
:members:
```

```{doxygenclass} samurai::MetadataReader
:members:
:undoc-members:
```
