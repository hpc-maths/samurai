# How-to: build a domain with obstacles

This guide shows you how to build a {{ project }} mesh around obstacles: holes cut out of a box, an obstacle described by a function, or a 3D object read from a surface file.
It ends with a finite volume demo that transports a solution around an obstacle.

## Before you start

- You know how to define a domain from boxes with `samurai::DomainBuilder`: see the [box how-to guide](box.md).
- You know the mesh types and their levels: see the [mesh how-to guide](mesh.md).
- Your program builds against {{ project }}: see the [CMake how-to guide](cmake.md).

Pick the method from the shape of your obstacles:

| Obstacles | Method | Dimension | Extra dependency |
| --- | --- | --- | --- |
| Axis-aligned boxes | `samurai::DomainBuilder::remove` | 2D or 3D | None |
| Any shape given by a function of the coordinates | Cell list filtered by the function | 2D or 3D | None |
| Closed surface in a file (OBJ) | `samurai::from_geometry` | 3D only | CGAL |

## Cut box-shaped obstacles

Call `remove` on a `samurai::DomainBuilder` once per obstacle, then pass the builder to `samurai::mra::make_mesh`.
This program cuts two square obstacles of side 0.25 out of a $2 \times 1$ channel and prints the area its cells cover:

```{literalinclude} snippet/geometry/two_holes.cpp
  :language: c++
```

With `-DBUILD_SNIPPETS=ON`, CMake builds it as the `two_holes` target.
It prints:

```{literalinclude} snippet/geometry/two_holes_output.txt
  :language: text
```

The covered area is the area of the channel minus the two obstacles, $2 - 2 \times 0.25^2 = 1.875$.
{{ project }} derives the cell length from the boxes: here the length at level 0 is 0.25, so a cell at the maximum level 5 has a side of $0.25 / 2^5 = 0.0078125$.

A `DomainBuilder` mesh cannot be periodic and cannot be built with MPI.
The [box how-to guide](box.md) lists these limits and the minimum size of a hole.

## Carve an obstacle from a level-set function

When the obstacle is not a box, describe it with a function of the coordinates, mesh its bounding box, and keep the cells whose center is outside the obstacle.
The example below uses the level-set function $\phi$ of a disc of radius 0.2 centered at $(0.5, 0.5)$: $\phi < 0$ in the obstacle and $\phi > 0$ in the fluid.
A function that returns `true` outside the obstacle works the same way.

1. Define the function:

   ```{literalinclude} snippet/geometry/disc_obstacle.cpp
     :language: c++
     :start-at: auto phi
     :end-at: };
     :dedent:
   ```

2. Mesh the bounding box with `samurai::mra::make_mesh`.
   Every cell of this first mesh is at the maximum level.

   ```{literalinclude} snippet/geometry/disc_obstacle.cpp
     :language: c++
     :start-at: samurai::Box<double, dim> box
     :end-at: auto mesh
     :dedent:
   ```

3. Copy the cells whose center is in the fluid into a cell list.
   Give the cell list the origin and the scaling factor of the first mesh, so that the integer coordinates of the cells keep their meaning.

   ```{literalinclude} snippet/geometry/disc_obstacle.cpp
     :language: c++
     :start-at: using mesh_t
     :end-before: // 3.
     :dedent:
   ```

4. Build the mesh from the cell list, with the same configuration:

   ```{literalinclude} snippet/geometry/disc_obstacle.cpp
     :language: c++
     :start-at: mesh = samurai::mra::make_mesh(cl, config);
     :end-at: mesh = samurai::mra::make_mesh(cl, config);
     :dedent:
   ```

The domain of the new mesh is the set of cells you kept, so the boundary conditions of a field apply on the edge of the obstacle as they do on the edge of the box.
The obstacle edge is a staircase of cells at the maximum level: raise `max_level` to follow the shape more closely.

`make_mesh` does not check the size of an obstacle carved this way, unlike `DomainBuilder` (see the troubleshooting section of the [box how-to guide](box.md)).

### Check the covered area

The full program is in `docs/source/howto/snippet/geometry/disc_obstacle.cpp` (target `disc_obstacle`).
After building the mesh, it compares the area of the cells with the exact area $2 - \pi \times 0.2^2$:

```{literalinclude} snippet/geometry/disc_obstacle.cpp
  :language: c++
  :start-at: using mesh_id_t
  :end-before: // 4.
  :dedent:
```

```{literalinclude} snippet/geometry/disc_obstacle_output.txt
  :language: text
  :end-before: level 2:
```

The cells cover the channel minus the disc to 0.12 %, with cells of side $1/64$ at level 6.

### Adapt the mesh around the obstacle

The program then gives a field `u` the value $e^{-50 \phi}$, which varies fast near the disc, attaches a Neumann boundary condition, and adapts the mesh with multiresolution (see the [adaptation how-to guide](adapt.md)):

```{literalinclude} snippet/geometry/disc_obstacle.cpp
  :language: c++
  :start-at: auto u_init
  :end-at: adapt(samurai::mra_config
  :dedent:
```

It prints the number of cells at each level:

```{literalinclude} snippet/geometry/disc_obstacle_output.txt
  :language: text
  :start-at: level 2:
```

The 1368 cells of the adapted mesh still cover an area of 1.87207: no coarse cell reaches into the obstacle.
The program saves `u` and the level of each cell in `disc_obstacle.h5`; this is the mesh of that file:

```{diagram}
:figure: disc_obstacle_mesh

The program keeps a level 6 cell when its center lies outside the disc, so the edge of the hole is a staircase of level 6 cells: the exact circle, in red, runs across the kept cells of the staircase and through the hole, never across a coarser cell.
Level 6 cells are hatched in red; away from the disc, adaptation merges cells down to level 2.
Four corner marks on the channel frame the square [0.25, 0.75) × [0.25, 0.75) around the disc, and the view under the channel enlarges it.
```

## Run a scheme around an obstacle

The demo `demos/FiniteVolume/linear_convection_obstacle.cpp` transports a square of $u = 1$ with the velocity $(1, -1)$ on the square $[-1, 1]^2$ with a hole $[0, 0.4]^2$.
It uses a fifth-order WENO convection scheme, an SSP-RK3 time scheme and multiresolution adaptation between levels 1 and 3.
The velocity field has a zero Dirichlet condition, so the boundary of the box and the edge of the hole are walls.

1. Build the demo:

   ```bash
   cmake -S . -B build -DBUILD_DEMOS=ON
   cmake --build build \
       --target finite-volume-linear-convection-obstacle
   ```

2. Run it until $t = 3$, the default final time, and save 12 snapshots:

   ```bash
   ./build/demos/FiniteVolume/finite-volume-linear-convection-obstacle \
       --nfiles 12
   ```

   It prints one line per time step and ends with:

   ```text
   iteration 504: t = 3.00, dt = 0.0059375
   iteration 505: t = 3.00, dt = 0.0015625000000208545
   ```

3. Open `linear_convection_obstacle_ite_<n>.xdmf`, from `_ite_0` (initial state) to `_ite_12` (final state), in ParaView.
   Each file holds `u` and the level of each cell.

To check the run, sum $u$ times the cell area over the cells of each snapshot: the result stays 0.25, the area of the initial square, in the 13 files, and the cells cover an area of 3.84, the box minus the hole.
No mass leaves through the walls: the square piles up against the edge of the hole on its way, then against the walls of the box: in the last snapshot all of it sits in the bottom-right corner cell, of side 0.0125, where $u = 0.25 / 0.0125^2 = 1600$.

## Build a 3D mesh from an OBJ surface

```{warning}
`mesh-from-obj` and `samurai::from_geometry` do not work on the current sources: they do not compile, and return an empty mesh once the build errors are patched ([#616](https://github.com/hpc-maths/samurai/issues/616)).
The rest of this section describes the intended use.
```

`samurai::from_geometry<3>` in `samurai/io/from_geometry.hpp` builds a 3D cell array from a closed triangulated surface.
It reads the file with CGAL, which triangulates the faces if needed, and meshes a box that encloses the surface, at level `start_level`.
Level by level up to `max_level`, it refines the cells that intersect a triangle of the surface.
By default it keeps only the cells that cut the surface, at `max_level`.
It returns a `samurai::CellArray<3>`, not a mesh:

| Argument | Meaning |
| --- | --- |
| `input_file` | Path of the surface file. CGAL picks the reader from the extension. |
| `start_level` | Level of the first uniform mesh of the enclosing box. |
| `max_level` | Level of the cells that cut the surface. |
| `keep_outside` | Also keep the cells whose center is outside the surface. Defaults to `false`. |
| `keep_inside` | Also keep the cells whose center is inside the surface. Defaults to `false`. |

The demo `mesh-from-obj` calls it, applies `make_graduation` to the result, and saves the cells with their level.
It needs CGAL and Eigen 3: with conda, install `cgal-cpp` and `eigen=3.4` from conda-forge.
The demo's CMake file asks for Eigen 3.1.0 or later, and CMake rejects Eigen 5.

1. Configure with the demos and CGAL turned on, and build the target:

   ```bash
   cmake -S . -B build -DBUILD_DEMOS=ON -DWITH_CGAL=ON
   cmake --build build --target mesh-from-obj
   ```

2. Run it on one of the two surfaces shipped with the demo, `apple.obj` or `human.obj`:

   ```bash
   ./build/demos/from_obj/mesh-from-obj \
       --input demos/from_obj/apple.obj --max-level 6
   ```

   | Option | Default | Effect |
   | --- | --- | --- |
   | `--input` | (required) | Surface file. |
   | `--init-level` | `1` | `start_level` of `from_geometry`. |
   | `--max-level` | `8` | `max_level` of `from_geometry`. |
   | `--keep-inside` | off | Keep the cells inside the object. |
   | `--keep-outside` | off | Keep the cells outside the object. |
   | `--path` | current directory | Output directory. |

3. Open `mesh_apple.xdmf` (the stem of the input file, prefixed with `mesh_`) in ParaView and color it by `level`.

## Related

- [Box how-to guide](box.md): domains made of boxes, the limits of `DomainBuilder`, and the minimum size of a hole.
- [Mesh how-to guide](mesh.md): mesh types, levels and configuration.
- [Adaptation how-to guide](adapt.md): multiresolution adaptation.
- [Plot how-to guide](plot.md): open the HDF5 and XDMF files in ParaView or Matplotlib.
