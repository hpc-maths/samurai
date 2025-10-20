# How-to: save your samurai mesh and fields

In this how-to guide, we will show you how to save your samurai mesh and fields to files. Saving meshes and fields is essential for post-processing, visualization, and restarting simulations. The data are saved in HDF5 format and a xdmf file is also created for visualization with Paraview.

We have two kinds of saving functions:
- `save` to save the data for post-processing purposes.
- `dump`and `load` to save and load the data for restarting simulations.

## Saving for post-processing

To save a samurai mesh and fields for post-processing, you can use the `save` function provided by samurai. Here is a simple example of how to save a mesh:

```{literalinclude} snippet/save/save_mesh.cpp
    :language: c++
```

In this example, we first create a 2D multi-resolution mesh over the box defined from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a minimum refinement level of 2 and a maximum refinement level of 5. Then, we use the `save` function to save the mesh to a file named "mesh_filename.h5". An accompanying "mesh_filename.xdmf" file is also created for visualization in Paraview.

If the filename is only provided, the file will be saved in the current directory as described in the snippet above in the comment.

```{note}
To have the `save` function available, you need to include the header file `samurai/io/hdf5.hpp`.
```

You can also save fields along with the mesh. Here is an example of how to save a scalar field and a vector field on the mesh:

```{literalinclude} snippet/save/save_field.cpp
    :language: c++
```

In this example, we create a scalar field named "u" and a vector field named "v" with 3 components on the multi-resolution mesh. We then use the `save` function to save both fields along with the mesh to a file named "output_path/fields.h5". An accompanying "output_path/fields.xdmf" file is also created for visualization in Paraview.

```{note}
- The first argument after the filename in the `save` function must be the mesh.
- You can save multiple fields at once by passing them as additional arguments to the `save` function.
```

```{caution}
We don't verify that the fields belong to the mesh provided. It is your responsibility to ensure that the fields are defined on the same mesh.
```

A default option exists to save some debug fields such as the level, the coordinates, and the indexes of each cell. You can enable this option in command line by using `--save-debug-fields`.

## Saving for restarting simulations

To save a samurai mesh and fields for restarting simulations, you can use the `dump` function provided by samurai. Here is a simple example of how to dump a mesh:

```{literalinclude} snippet/save/dump_mesh.cpp
    :language: c++
```

In this example, we first create a 2D multi-resolution mesh over the box defined from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a minimum refinement level of 2 and a maximum refinement level of 5. Then, we use the `dump` function to save the mesh to a file named "restart_file.h5". And, we use the `load` function to load the mesh from the file. The arguments are the same as before with the `save` function.

```{caution}
If you use `dump` in a parallel MPI program, the `load` function must be called with the same number of MPI processes as when the `dump` function was called.
```

## Save all the sub-meshes

In samurai, a mesh can be composed of several sub-meshes. By default, the `save` functions only save the main mesh.

For example, a `UniformMesh`is composed by the true cells mesh and a ghost cells mesh at the boundary of the domain. In the same way, a `MRMesh` is composed of several sub-meshes to compute the details.

For debugging or post-processing purposes, it can be interesting to save all the sub-meshes of a mesh. To do so, you can add some boolean flags in the `save` function:

```{literalinclude} snippet/save/save_all_submeshes.cpp
    :language: c++
```

We added two boolean flags after the filename in the `save` function:

- The first flag indicates if we want to save the mesh by levels. If set to `true`, each level of the mesh will be saved as a separate sub-mesh in the HDF5 file.
- The second flag indicates if we want to save the mesh by ids. If set to `true`, each sub-mesh identified by its unique ID will be saved as a separate sub-mesh in the HDF5 file.

```{caution}
You always have to provide a path and a filename when you want to save all the sub-meshes.
```

To plot your data, please refer to the [plot how-to guide](plot.md).
