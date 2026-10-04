# Meshes and cell arrays

This page documents the meshes of {{ project }} and the cell collections they are made of.
`samurai::UniformMesh` holds the cells of a single level.
The multiresolution mesh `samurai::MRMesh` and the AMR mesh `samurai::amr::Mesh` hold cells between a minimum and a maximum level; they are described by a `samurai::mesh_config` and built by `samurai::mra::make_mesh` and `samurai::amr::make_mesh`.
Each mesh stores one `samurai::CellArray` (one `samurai::LevelCellArray` per level) per mesh id; `samurai::CellList` is the mutable collection you fill before you freeze it into a cell array.
For how to build each mesh type, see the {doc}`mesh how-to <../howto/mesh>`.

## Mesh configuration

`samurai::mesh_config` describes a multiresolution or AMR mesh: the dimension and the prediction stencil radius as template arguments, then the levels, the stencil radius, the graduation width and the periodicity through chained setters.
`make_mesh` calls `parse_args()` on its copy of the configuration, so the command-line options replace the values set in the code unless `disable_args_parse()` was called.

```{doxygenclass} samurai::mesh_config
:members:
:undoc-members:
```

## Uniform mesh

`samurai::UniformMesh` covers a box with the cells of one level.
Its configuration is `samurai::UniformConfig`, whose second template argument is the ghost width.
Its mesh ids are `cells` and `cells_and_ghosts`; `reference` is `cells_and_ghosts`.

```{doxygenstruct} samurai::UniformConfig
:members:
:undoc-members:
```

```{doxygenenum} samurai::UniformMeshId
```

```{doxygenclass} samurai::UniformMesh
:members: UniformMesh, nb_cells, operator[], get_interval, get_index, origin_point, set_origin_point, scaling_factor, set_scaling_factor, cell_length, cfg, to_stream
:undoc-members:
```

## Multiresolution mesh

`samurai::mra::make_mesh` builds a `samurai::MRMesh` from a box, a `samurai::DomainBuilder`, a cell list or a cell array, and a `samurai::mesh_config`.
It sets the start level to the max level: multiresolution needs the solution on the finest level before it can coarsen the mesh.
`samurai::mra::make_empty_mesh` builds a mesh with no cells that holds the configuration, to be filled later by assignment or by `samurai::load`.

```{doxygenfunction} samurai::mra::make_mesh(const samurai::Box<double, mesh_config_t::dim> &b, const mesh_config_t &cfg)
```

```{doxygenfunction} samurai::mra::make_mesh(const samurai::DomainBuilder<mesh_config_t::dim> &domain_builder, const mesh_config_t &cfg)
```

```{doxygenfunction} samurai::mra::make_empty_mesh
```

The mesh ids of `samurai::MRMesh` are listed by `samurai::MRMeshId`.
The `reference` id, on which the fields are sized, holds all the cells of the mesh.

```{doxygenenum} samurai::MRMeshId
```

The constructors that take `min_level` and `max_level` instead of a `samurai::mesh_config` are deprecated.

```{doxygenclass} samurai::MRMesh
:members: MRMesh, exists
:undoc-members:
```

## AMR mesh

`samurai::amr::make_mesh` builds a `samurai::amr::Mesh` from a box, a cell list or a cell array, and a `samurai::mesh_config`.
The cells start at the start level of the configuration, which is the max level when `start_level` is not set.
`samurai::amr::make_empty_mesh` builds a mesh with no cells that holds the configuration.

```{doxygenfunction} samurai::amr::make_mesh(const samurai::Box<double, mesh_config_t::dim> &b, const mesh_config_t &cfg)
```

```{doxygenfunction} samurai::amr::make_empty_mesh
```

```{doxygenenum} samurai::amr::AMR_Id
```

```{doxygenclass} samurai::amr::Mesh
:members: Mesh
:undoc-members:
```

## Common mesh interface

`samurai::MRMesh` and `samurai::amr::Mesh` derive from `samurai::Mesh_base`, which holds the cell arrays of every mesh id and the levels, the domain and the periodicity of the mesh.
`mesh[id]` returns the cell array of the mesh id `id`; `nb_cells()` counts the cells of the `reference` id.

```{doxygenclass} samurai::Mesh_base
:members: nb_cells, operator[], min_level, max_level, graduation_width, ghost_width, max_stencil_radius, cfg, origin_point, scaling_factor, cell_length, min_cell_length, domain, subdomain, get_union, is_periodic, periodicity, get_interval, get_index, get_cell, to_stream
:undoc-members:
```

## Cell arrays

`samurai::LevelCellArray` holds the cells of one level as a sorted, immutable set of intervals: one array of intervals per direction, with the offsets that link an interval along `y` to its intervals along `x`.
It is the unit of the set algebra.
`samurai::CellArray` holds one `LevelCellArray` per level.

```{doxygenclass} samurai::CellArray
:members: CellArray, operator[], get_interval, get_index, get_cell, nb_cells, empty, min_level, max_level, origin_point, scaling_factor, cell_length, update_index, to_stream
:undoc-members:
```

```{doxygenclass} samurai::LevelCellArray
:members: LevelCellArray, get_interval, get_index, get_cell, empty, nb_intervals, nb_cells, level, cell_length, min_indices, max_indices, minmax_indices, min_corner, max_corner, origin_point, scaling_factor, to_stream
:undoc-members:
```

## Cell lists

`samurai::CellList` is the mutable, build-time counterpart of `samurai::CellArray`: one `samurai::LevelCellList` per level, into which you add intervals or cells in any order.
Build a `CellArray` or a `LevelCellArray` from it to get the sorted intervals.

```{doxygenclass} samurai::CellList
:members: CellList, operator[], origin_point, scaling_factor, clear
:undoc-members:
```

```{doxygenclass} samurai::LevelCellList
:members: LevelCellList, operator[], add_cell, level, empty, clear
:undoc-members:
```
