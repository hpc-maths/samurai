# How-to: create a samurai mesh

This guide shows you how to create the three mesh types samurai provides, and how to set their levels and ghost cells.

Before you start, you need:

- samurai installed (see the [installation guide](installation.md));
- a domain to build the mesh on (see the [box how-to guide](box.md)).

samurai provides three mesh types:

- **Uniform mesh** ({cpp:class}`samurai::UniformMesh`): all cells have the same size, at a single level.
- **Multiresolution mesh** (`samurai::mra::make_mesh`): cells can have different sizes, between a minimum and a maximum level, and multiresolution adapts the mesh to the solution.
- **Adaptive mesh refinement (AMR) mesh** (`samurai::amr::make_mesh`): cells can have different sizes, and you refine or coarsen them with your own criterion.

To choose between multiresolution and AMR, read {ref}`philosophy-mr-amr`: it explains how each one decides where the cells go.

## Create a uniform mesh

Build a `samurai::UniformMesh` from a box and a level:

```{literalinclude} snippet/mesh/uniform.cpp
  :language: c++
  :end-before: // Print the number
```

This example creates a 2D uniform mesh on the box from $(0.0, 0.0)$ to $(1.0, 1.0)$ at level 4, which gives $2^4$ cells along each dimension.

The second template parameter of `samurai::UniformConfig` sets the number of ghost cells around the domain.
It is 1 by default.
For example, to get 2 ghost cells:

```cpp
using config_t = samurai::UniformConfig<dim, 2>; // 2 ghost cells
```

## Create a multiresolution mesh

Describe the mesh with {cpp:class}`samurai::mesh_config`, then pass it to `samurai::mra::make_mesh` with the box:

```{literalinclude} snippet/mesh/mrmesh.cpp
  :language: c++
  :end-before: // Print the number
```

This example creates a 2D multiresolution mesh on the box from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a minimum level of 2 and a maximum level of 5.
All cells start at the maximum level, because multiresolution needs the solution at the finest level before it can coarsen the mesh.
The {doc}`adaptation how-to guide <adapt>` shows how to adapt it to your solution from there.

## Create an AMR mesh

Use the same `samurai::mesh_config` and pass it to `samurai::amr::make_mesh`.
Set the level the cells start at with `start_level`:

```{literalinclude} snippet/mesh/amrmesh.cpp
  :language: c++
  :end-before: // Print the number
```

This example creates a 2D AMR mesh on the box from $(0.0, 0.0)$ to $(1.0, 1.0)$ with all cells at level 4, a minimum level of 2 and a maximum level of 5.
Without `start_level`, the cells start at the maximum level.
The start level must lie between the minimum and the maximum level, otherwise building the mesh throws `std::invalid_argument`.

## Set the ghost cells and the graduation

Multiresolution and AMR meshes take their stencil and graduation settings from `samurai::mesh_config`.
Call the setters you need on the configuration before you build the mesh:

```{literalinclude} snippet/mesh/ghost_config.cpp
  :language: c++
  :start-at: // Prediction stencil radius
  :end-at: auto mesh =
  :dedent:
```

Choose the values from your numerical scheme:

- `max_stencil_radius` is the number of neighbors your scheme reads in each direction. It is 2 by default: samurai raises any smaller value to 2 when it builds the mesh, as explained below.
- `graduation_width` is the width, in cells, of the neighborhood of a cell in which levels differ by at most one. It is 1 by default.
- The second template parameter of `mesh_config` is the radius of the stencil of the prediction operator. It is 1 by default, which gives a prediction of order 3 ($2s+1$ with $s = 1$).

The ghost width is the larger of the stencil radius and the prediction stencil radius.
When it builds the mesh, samurai raises the stencil radius to at least 2, which leaves enough ghost cells at the boundary for reconstruction and for transfers between levels.
The default ghost width is therefore 2.
To keep a stencil radius of 1, call `disable_minimal_ghost_width()` on the configuration.

## Override the configuration from the command line

When your program calls `samurai::initialize(argc, argv)` before it creates the mesh, these command-line options replace the values set in the code:

| Option | Replaces |
| --- | --- |
| `--min-level` | `min_level` |
| `--max-level` | `max_level` |
| `--start-level` | `start_level` (AMR mesh) |
| `--max-stencil-radius` | `max_stencil_radius` |
| `--graduation-width` | `graduation_width` |

For example, `./your_program --max-level 7` builds the mesh up to level 7 whatever `max_level` says in the code.
To keep the code values, call `disable_args_parse()` on the configuration.
The [options how-to guide](options.md) explains how to parse and list these options.

## Check the mesh

The complete programs of the examples above print the number of cells of the mesh.
For the uniform mesh, `docs/source/howto/snippet/mesh/uniform.cpp` ends with:

```{literalinclude} snippet/mesh/uniform.cpp
  :language: c++
  :start-at: // Print the number
  :end-at: std::endl;
  :dedent:
```

It prints the $16 \times 16$ cells at level 4, and the $18 \times 18$ cells once the layer of one ghost cell is added:

```{literalinclude} snippet/mesh/uniform_output.txt
  :language: text
```

The ghost cells surround the box on every side, corners included:

```{diagram}
:figure: ghost_ring

The level-4 mesh of `uniform.cpp`: 256 cells inside the heavy outline of the box, and a ring of ghost cells, dashed, around it.
The ring is one cell wide because `samurai::UniformConfig<dim>` keeps its default ghost width, 1.
Each row of cells is widened by one ghost at both ends and copied to the rows just above and below it, so the four corner ghosts are part of the ring and of the count: $18 \times 18 = 324$, that is 256 cells and 68 ghosts.
Under the mesh, the interval of a row along x, without and with the ghosts.
```

The multiresolution and AMR programs count the cells of each level with `mesh.nb_cells(level, mesh_id_t::cells)`:

```{literalinclude} snippet/mesh/mrmesh.cpp
  :language: c++
  :start-at: // Print the number
  :end-at: "}"
  :dedent:
```

The multiresolution mesh has all its cells at level 5:

```{literalinclude} snippet/mesh/mrmesh_output.txt
  :language: text
```

The AMR mesh has all its cells at level 4, its start level:

```{literalinclude} snippet/mesh/amrmesh_output.txt
  :language: text
```

The program `docs/source/howto/snippet/mesh/ghost_config.cpp` of the previous section prints the ghost width of its mesh and of the same mesh with the default settings:

```{literalinclude} snippet/mesh/ghost_config_output.txt
  :language: text
```

To see every sub-mesh (`cells`, `cells and ghosts`, ...) interval by interval, print the mesh itself with `std::cout << mesh`.
Its lines are long: one line holds all the intervals of a level along one direction.

## Next steps

- Create fields on the mesh with the [field how-to guide](field.md).
- Loop over the cells of the mesh with the [loop how-to guide](loop.md).
- Save the mesh for visualization with the [save how-to guide](save.md).
- Adapt a multiresolution mesh to your solution with the {doc}`adaptation how-to guide <adapt>`.
- Look up the mesh classes and the members of `samurai::mesh_config` in the {doc}`mesh API reference <../api/mesh>`.
