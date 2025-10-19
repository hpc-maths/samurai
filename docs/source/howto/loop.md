# How-to: loop over cells in a samurai mesh

In this how-to guide, we will show you how to loop over the cells of a samurai mesh. Looping over cells is a common operation when you want to perform computations or apply algorithms on each cell of the mesh.

To follow this guide, you should already have a samurai mesh created. If you don't know how to create a mesh, please refer to the mesh how-to guide [here](mesh.md).

It can be also interesting to have a samurai field defined on the mesh. If you don't know how to create a field, please refer to the field how-to guide [here](field.md).

## Looping over cells

To loop over the cells of a samurai mesh, you can use the `for_each_cell` function provided by samurai. Here is a simple example of how to loop over the cells of a multi-resolution mesh and print the level and index of each cell:

```{literalinclude} snippet/loop/for_each_cell_mesh.cpp
  :language: c++
```

In this example, we first create a 2D multi-resolution mesh over the box defined from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a minimum refinement level of 2 and a maximum refinement level of 5. Then, we use the `for_each_cell` function to loop over each cell in the mesh. Inside the loop, we print the level and center of each cell.

You can also access and modify the values of a samurai field while looping over the cells. Here is an example of how to set the values of a scalar field to the level of each cell:

```{literalinclude} snippet/loop/for_each_cell_field.cpp
  :language: c++
```

In this example, we create a scalar field named "u" on the multi-resolution mesh. Inside the loop, we compute the center coordinates of each cell and set the value of the field at that cell to a Gaussian function centered at (0.5, 0.5).

## Looping over intervals

Another way to iterate over the cells of a samurai mesh is to use intervals. You can use the `for_each_interval` function to loop over intervals. Here is an example:

```{literalinclude} snippet/loop/for_each_interval_mesh.cpp
  :language: c++
```

In this example, we use the `for_each_interval` function to loop over each interval in the mesh. Inside the loop, we print the interval, level, and index of each interval. `index` is an array of size $dim-1$ that contains the indices of the other dimensions ($y$, $z$, ...). If you want to deeper understand intervals, please refer to the interval tutorial [here](../tutorial/interval.rst)

Let's see how to set the values of a scalar field using intervals:

```{literalinclude} snippet/loop/for_each_interval_field.cpp
  :language: c++
```
