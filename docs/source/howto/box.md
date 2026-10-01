# How-to: create a samurai domain from boxes

A samurai domain is the region of space your mesh covers.
This guide shows how to define it with boxes: one box for a rectangle or a cuboid, several boxes joined together for an L-shaped or T-shaped domain, and boxes cut out of it for holes.

## Before you start

- samurai is installed: see the [installation guide](installation.md).
- Your program builds against samurai: see the [CMake guide](cmake.md).
- The domain is made of axis-aligned boxes.
  For a domain read from a surface file (OBJ), see the geometry links at the end of this page.

## Create a single box

A `samurai::Box<double, dim>` takes its lower corner and its upper corner.
This 3D box goes from $(0, 0, 0)$ to $(1, 1, 1)$:

```{literalinclude} snippet/box/3d_box.cpp
  :language: c++
```

A 2D box works the same way with two coordinates per corner.
This one goes from $(-1, -1)$ to $(1, 1)$:

```{literalinclude} snippet/box/2d_box.cpp
  :language: c++
```

A single box is enough to build any samurai mesh: see the [mesh guide](mesh.md).

## Join several boxes

To build a domain that is the union of several boxes, use a `samurai::DomainBuilder<dim>`.
Its constructor takes the first box, and each call to `add` joins another one.
The boxes can touch or overlap.

This L-shaped domain is a $2 \times 1$ rectangle with a $1 \times 1$ square on top of its left half:

```c++
#include <samurai/domain_builder.hpp>

int main()
{
    static constexpr std::size_t dim = 2;

    samurai::DomainBuilder<dim> domain({0.0, 0.0}, {2.0, 1.0});
    domain.add({0.0, 1.0}, {1.0, 2.0});

    return 0;
}
```

`add` and the constructor accept either two corners or a `samurai::Box<double, dim>`.

## Cut holes

To remove a box from the domain, call `remove` on the `DomainBuilder`.
This domain is the square from $(-1, -1)$ to $(1, 1)$ with a hole from $(0, 0)$ to $(0.4, 0.4)$:

```{literalinclude} snippet/box/2d_box_with_hole.cpp
  :language: c++
```

samurai applies every `remove` after every `add`, whatever the order of the calls.
A box added inside a hole therefore stays removed.

## Build the mesh

Pass the `DomainBuilder` to `samurai::mra::make_mesh`, which builds a multiresolution mesh (`MRMesh`).
This excerpt from the [linear convection with obstacle demo](https://github.com/hpc-maths/samurai/blob/main/demos/FiniteVolume/linear_convection_obstacle.cpp) builds the mesh of the domain with a hole:

```{literalinclude} ../../../demos/FiniteVolume/linear_convection_obstacle.cpp
  :language: c++
  :start-at: samurai::DomainBuilder<dim> domain
  :end-at: make_mesh(domain, config);
  :dedent: 4
```

`max_stencil_size(6)` sets the stencil size that the WENO5 scheme of the demo needs.
The stencil also sets the minimum size of a hole: see the troubleshooting section below.

A `DomainBuilder` has these limits:

- Only `samurai::mra::make_mesh` accepts it. `samurai::amr::make_mesh` and `samurai::UniformMesh` take a single `samurai::Box`.
- The mesh cannot be periodic: `make_mesh` throws `std::runtime_error` with the message `Periodicity is not implemented with DomainBuilder.`
- The mesh cannot be created when samurai is built with MPI (`SAMURAI_WITH_MPI`): `make_mesh` throws `std::runtime_error` with the message `MPI is not implemented with DomainBuilder.`

## Check the result

Build the mesh of the L-shaped domain, print its number of cells and save it:

```c++
#include <iostream>

#include <samurai/domain_builder.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::DomainBuilder<dim> domain({0.0, 0.0}, {2.0, 1.0});
    domain.add({0.0, 1.0}, {1.0, 2.0});

    auto config = samurai::mesh_config<dim>().min_level(2).max_level(4);
    auto mesh   = samurai::mra::make_mesh(domain, config);

    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    std::cout << mesh.nb_cells(mesh_id_t::cells) << std::endl;

    samurai::save("l_shaped_domain", mesh);

    samurai::finalize();
    return 0;
}
```

The program prints `768`.
`make_mesh` creates every cell at the maximum level, 4 here, so cells have a side of $2^{-4}$ and the domain of area 3 holds $3 \times 4^4 = 768$ of them.

Open `l_shaped_domain.xdmf` in ParaView or VisIt: you see the L shape.
The [save guide](save.md) describes the output files.

## Troubleshooting

### `The hole ... is too small to apply the BC at level ...`

samurai places ghost cells inside each hole to apply the boundary conditions.
Each side of a hole must therefore be at least $2 r$ cells of the minimum level long, where $r$ is the stencil radius set in the mesh configuration (`max_stencil_radius`, or half of `max_stencil_size`, rounded up).
By default samurai lowers the scaling factor (the cell length at level 0) until every hole is large enough.
If you set `scaling_factor` in the mesh configuration, samurai keeps your value and stops with this message instead.
Lower `scaling_factor`, raise `min_level`, or make the hole larger.

## Related

- [Mesh guide](mesh.md): uniform, multiresolution and AMR meshes.
- [Save guide](save.md): write the mesh and its fields to HDF5.
- {doc}`Box class reference <../api/box>`.
- Geometries beyond boxes:
  - the [mesh from OBJ demo](https://github.com/hpc-maths/samurai/blob/main/demos/from_obj/main.cpp) builds a 3D mesh from a surface file with `samurai::from_geometry` (header [`samurai/io/from_geometry.hpp`](https://github.com/hpc-maths/samurai/blob/main/include/samurai/io/from_geometry.hpp)); it needs CGAL and the CMake option `WITH_CGAL`;
  - the [linear convection with obstacle demo](https://github.com/hpc-maths/samurai/blob/main/demos/FiniteVolume/linear_convection_obstacle.cpp) runs a finite volume scheme on a domain with a hole.
