# How-to: create a samurai domain using boxes

In this how-to guide, we will show you how to create a samurai domain using boxes. Boxes are a simple way to define a rectangular domain in samurai. You can create complex domains by combining multiple boxes and make differences between them.

First, make sure you have samurai installed. If you haven't installed it yet, please refer to the installation guide [here](installation.md).

## A simple example

Let's start with a simple example of creating a domain using one box.

```{literalinclude} snippet/box/3d_box.cpp
  :language: c++
```

In this example, we create a 3D box with the lower corner at $(0.0, 0.0, 0.0)$ and the upper corner at $(1.0, 1.0, 1.0)$.

If you want to create a 2D box, you can do it following the same principle as this:

```{literalinclude} snippet/box/2d_box.cpp
  :language: c++
```

In this example, we create a 2D box with the lower corner at $(-1.0, -1.0)$ and the upper corner at $(1.0, 1.0)$.

That's it! You have successfully created a samurai domain using boxes. You can now use this domain to create meshes, fields, and perform simulations.

To create a mesh from the box, please refer to the mesh how-to guide [here](mesh.md).

## Combining multiple boxes

You can also create more complex domains by combining multiple boxes. For example, you can create a domain that is the union of two boxes:

```{literalinclude} snippet/box/2d_box_with_hole.cpp
  :language: c++
```

In this example, we first create a box from $(-1.0, -1.0)$ to $(1.0, 1.0)$, and then we remove a smaller box from $(0.0, 0.0)$ to $(0.4, 0.4)$. The `DomainBuilder` class allows you to easily add and remove boxes to create complex domains.
