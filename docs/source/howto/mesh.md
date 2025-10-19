# How-to: create a samurai mesh

Various mesh types are available in samurai. This how-to guide will show you how to create a simple mesh.

Before creating a mesh, make sure you have samurai installed. If you haven't installed it yet, please refer to the installation guide [here](installation.md).

You also need to create a domain before creating a mesh. If you don't know how to create a domain, please refer to the box how-to guide [here](box.md).

The default mesh types available in samurai are:

- **UniformMesh**: A uniform mesh where all the cells have the same size.
- **MRMesh**: A multi-resolution mesh where the cells can have different sizes based on a refinement level.
- **AMRMesh**: An adaptive mesh refinement mesh where the cells can be refined or coarsened based on a given criterion.

## Creating a UniformMesh

To create a uniform mesh, you can use the `UniformMesh` class. Here is a simple example:

```{literalinclude} snippet/mesh/uniform.cpp
  :language: c++
```

In this example, we create a 2D uniform mesh over the box defined from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a refinement level of 4 which means that the mesh will have $2^4$ cells along each dimension.

The `UniformConfig` class allows you to specify the number of ghost cells you want. By default, there are one ghost cells all around the domain. If you want to change this, you can do it like this:

```cpp
using config_t = samurai::UniformConfig<dim, 2>; // 2 ghost cells
```

## Creating a multi-resolution mesh

To create a multi-resolution mesh, you can use the `MRMesh` class. Here is a simple example:

```{literalinclude} snippet/mesh/mrmesh.cpp
  :language: c++
```

In this example, we create a 2D multi-resolution mesh over the box defined from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a minimum refinement level of 2 and a maximum refinement level of 5. At the initialization, all cells are created at the maximum level. This is due to the multi-resolution constraint that requires that the solution is known at the finest level at the beginning.

The `MRConfig` class allows you to specify the number of ghost cells you want for your numerical scheme, the number of cells for the graduation of the mesh and the number of ghost cells used by the prediction operator. The prediction operator is used to compute the details defined in the multi-resolution process in order to adapt the mesh accordingly. This operator is also used to compute the solution at a finest level (when you refine a cell for example).

By default, there are

- the stencil of the numerical scheme is one
- the graduation is set to one cell
- the stencil of prediction operator is one which corresponds to a prediction of order 3 ($2s+1$).

You can change these parameters like this:

```cpp
using config_t = samurai::MRConfig<dim, 2, 3, 2>; // 2 ghost cells, graduation of 3 cells, prediction stencil of 2
```

## Creating an AMR mesh

To create an adaptive mesh refinement mesh, you can use the `AMRMesh` class. Here is a simple example:

```{literalinclude} snippet/mesh/amrmesh.cpp
  :language: c++
```

In this example, we create a 2D adaptive mesh refinement mesh over the box defined from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a starting refinement level of 4, a minimum refinement level of 2, and a maximum refinement level of 5.
