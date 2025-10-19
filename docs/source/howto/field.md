# How-to: create a samurai field

Before creating a samurai field, you have to create a mesh. If you don't know how to create a mesh, please refer to the mesh how-to guide [here](mesh.md).

A samurai field is a data structure that allows you to store and manipulate data on a samurai mesh. Fields can be defined on different mesh types, such as uniform meshes, multi-resolution meshes, and adaptive meshes.

Two types of fields are available in samurai:

- **scalar fields**: Fields that store a single value per cell.
- **vector fields**: Fields that store multiple values per cell (e.g., a vector with several components).

## Creating a scalar field

To create a scalar field, you can use the `make_scalar_field` function. Here is a simple example of creating a scalar field on a multi-resolution mesh:

```{literalinclude} snippet/field/scalar_field.cpp
  :language: c++
```

In this example, we first create a 2D multi-resolution mesh over the box defined from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a minimum refinement level of 2 and a maximum refinement level of 5. Then, we create a scalar field named "u" on the mesh using the `make_scalar_field` function. The type of the field values is specified as `double`. You can replace `double` with any other data type as needed.

## Creating a vector field

To create a vector field, you can use the `make_vector_field` function. Here is a simple example of creating a vector field with 3 components on a multi-resolution mesh:

```{literalinclude} snippet/field/vector_field.cpp
  :language: c++
```

In this example, we create a vector field named "v" with 3 components on the same multi-resolution mesh. The type of the field values is specified as `double`. You can replace `double` with any other data type as needed, and change the number of components by modifying the second template parameter.