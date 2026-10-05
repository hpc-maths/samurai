# Graduation of a mesh

```{toctree}
:hidden:

graduation_case_1
graduation_case_2
graduation_case_3
```

```{definition} graded mesh
:label: def-graded-mesh

A mesh is graded when two neighboring cells differ by at most one level: a cell at level $l$ only touches cells at levels $l - 1$, $l$ and $l + 1$.
```

Adaptive mesh refinement needs this property, because the values of the ghost cells at level $l$ are reconstructed from the next coarser or the next finer level.

In this series of tutorials, we write the graduation ourselves with the set algebra of {{ project }}, on three 2D meshes built at random or from a criterion.
Each case comes with a demo in `demos/tutorial/` that we build and run.

## The three cases

- {doc}`Case 1 <graduation_case_1>`: the mesh has cells at several levels that do not overlap, and we make it graded.
- {doc}`Case 2 <graduation_case_2>`: the mesh has cells at several levels that overlap. We first remove the overlaps, then make the mesh graded.
- {doc}`Case 3 <graduation_case_3>`: the mesh is graded and a refinement criterion tags the cells to refine. We change the tags so that the refined mesh is still graded.

Case 2 reuses the algorithm of case 1, so read case 1 first.

## Before you start

- You have a build tree of {{ project }} configured from the sources, as described in {ref}`build-from-source`.
  The commands of each case run from the root of the repository and use the build directory `build`.
- You know how to build a {cpp:class}`samurai::CellArray` from a {cpp:class}`samurai::CellList` and how to write a subset with {cpp:func}`samurai::intersection` and {cpp:func}`samurai::translate`.
- To look at the meshes the demos write, you need a viewer for `.xdmf` files such as ParaView (see the {doc}`plot how-to guide <../howto/plot>`).

## The graduation functions of the library

These tutorials show how a graduation works.
In your own code, use the functions of `samurai/algorithm/graduation.hpp`:

- `samurai::make_graduation(ca)` grades a {cpp:class}`samurai::CellArray` in place. Its second argument is the graduation width, 1 by default.
- {cpp:func}`samurai::graduation` takes a tag field of an AMR mesh, filled with `samurai::CellFlag` values, and a stencil of directions. It changes the tags so that the mesh built from them is graded, as in case 3. The AMR demos, such as `demos/FiniteVolume/AMR_Burgers_Hat.cpp`, call it before each mesh update.
- {cpp:func}`samurai::is_graduated` returns `true` when a mesh or a cell array is graded along the directions of its stencil argument. The default stencil holds the axis directions only.

The multiresolution adaptation (`samurai::make_MRAdapt`) grades the mesh at each adaptation step, and {cpp:func}`samurai::graduation` does the same for AMR meshes.
Both read the graduation width of the mesh configuration (see the {doc}`mesh how-to guide <../howto/mesh>`).
