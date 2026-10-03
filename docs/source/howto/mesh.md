# How-to: create a samurai mesh

This guide shows you how to create the three mesh types samurai provides, and how to set their levels and ghost cells.

Before you start, you need:

- samurai installed (see the [installation guide](installation.md));
- a domain to build the mesh on (see the [box how-to guide](box.md)).

samurai provides three mesh types:

- **Uniform mesh** (`samurai::UniformMesh`): all cells have the same size, at a single level.
- **Multiresolution mesh** (`samurai::mra::make_mesh`): cells can have different sizes, between a minimum and a maximum level, and multiresolution adapts the mesh to the solution.
- **Adaptive mesh refinement (AMR) mesh** (`samurai::amr::make_mesh`): cells can have different sizes, and you refine or coarsen them with your own criterion.

## Create a uniform mesh

Build a `samurai::UniformMesh` from a box and a level:

```{literalinclude} snippet/mesh/uniform.cpp
  :language: c++
```

This example creates a 2D uniform mesh on the box from $(0.0, 0.0)$ to $(1.0, 1.0)$ at level 4, which gives $2^4$ cells along each dimension.

The second template parameter of `samurai::UniformConfig` sets the number of ghost cells around the domain.
It is 1 by default.
For example, to get 2 ghost cells:

```cpp
using config_t = samurai::UniformConfig<dim, 2>; // 2 ghost cells
```

## Create a multiresolution mesh

Describe the mesh with `samurai::mesh_config`, then pass it to `samurai::mra::make_mesh` with the box:

```{literalinclude} snippet/mesh/mrmesh.cpp
  :language: c++
```

This example creates a 2D multiresolution mesh on the box from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a minimum level of 2 and a maximum level of 5.
All cells start at the maximum level, because multiresolution needs the solution at the finest level before it can coarsen the mesh.

## Create an AMR mesh

Use the same `samurai::mesh_config` and pass it to `samurai::amr::make_mesh`.
Set the level the cells start at with `start_level`:

```{literalinclude} snippet/mesh/amrmesh.cpp
  :language: c++
```

This example creates a 2D AMR mesh on the box from $(0.0, 0.0)$ to $(1.0, 1.0)$ with all cells at level 4, a minimum level of 2 and a maximum level of 5.
Without `start_level`, the cells start at the maximum level.
The start level must lie between the minimum and the maximum level, otherwise building the mesh throws `std::invalid_argument`.

## Set the ghost cells and the graduation

Multiresolution and AMR meshes take their stencil and graduation settings from `samurai::mesh_config`.
Chain the setters you need:

```cpp
auto config = samurai::mesh_config<dim, 2>() // prediction stencil radius of 2
                  .min_level(2)
                  .max_level(5)
                  .max_stencil_radius(2)     // the numerical scheme reaches 2 cells in each direction
                  .graduation_width(3);      // levels differ by at most one within 3 cells
auto mesh   = samurai::mra::make_mesh(box, config);
```

Choose the values from your numerical scheme:

- `max_stencil_radius` is the number of neighbors your scheme reads in each direction. It is 1 by default.
- `graduation_width` is the width, in cells, of the neighborhood of a cell in which levels differ by at most one. It is 1 by default.
- The second template parameter of `mesh_config` is the radius of the stencil of the prediction operator. It is 1 by default, which gives a prediction of order 3 ($2s+1$ with $s = 1$).

The ghost width is the larger of the stencil radius and the prediction stencil radius.
samurai raises the stencil radius to at least 2, which leaves enough ghost cells at the boundary for reconstruction and for transfers between levels.
The default ghost width is therefore 2.
To keep a stencil radius of 1, call `disable_minimal_ghost_width()` on the configuration.

## Override the configuration from the command line

When your program calls `samurai::initialize(argc, argv)` before it creates the mesh, these command-line options replace the values set in the code:

| Option | Replaces |
|---|---|
| `--min-level` | `min_level` |
| `--max-level` | `max_level` |
| `--start-level` | `start_level` (AMR mesh) |
| `--max-stencil-radius` | `max_stencil_radius` |
| `--graduation-width` | `graduation_width` |

For example, `./your_program --max-level 7` builds the mesh up to level 7 whatever `max_level` says in the code.
To keep the code values, call `disable_args_parse()` on the configuration.
The [options how-to guide](options.md) explains how to parse and list these options.

## Check the mesh

Print the mesh to check its levels and cells:

```cpp
#include <iostream>

std::cout << mesh << std::endl;
```

The output lists each sub-mesh (`cells`, `cells and ghosts`, ...) with its cells level by level.
For the multiresolution example, the `cells` sub-mesh has cells at level 5 only; for the AMR example, at level 4 only.

## Next steps

- Create fields on the mesh with the [field how-to guide](field.md).
- Loop over the cells of the mesh with the [loop how-to guide](loop.md).
- Save the mesh for visualization with the [save how-to guide](save.md).
