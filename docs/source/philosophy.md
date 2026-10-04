# Philosophy of {{ project }}

{{ project }} is a C++ library for adaptive mesh methods on Cartesian grids.
Its name says how it works: Structured Adaptive mesh and MUlti-Resolution based on Algebra of Intervals.
Instead of a tree of cells, it stores each level of the mesh as sets of integer intervals, and it expresses most operations on the mesh as operations of a set algebra over those intervals.

This page explains why the library is built this way, what the design gives you, and what it costs.
It is meant for readers who want to judge whether {{ project }} fits their problem, and for contributors who want to understand the reasons behind the code.
For a hands-on introduction to the data structure, start with the {doc}`interval tutorial <tutorial/interval>`.

## The problem: neighbors on an adaptive mesh

Adaptive methods put fine cells where the solution varies and coarse cells elsewhere.
Most data structures built for this purpose are trees: a quadtree in 2D, an octree in 3D.
Adding or removing a cell is cheap in a tree once you know where it goes.
Finding a given cell, on the other hand, means walking down the tree from its root.

Finding cells is what a numerical scheme does all the time.
A finite volume or finite element scheme evaluated on a cell reads the cells of its stencil around it, so every cell of the mesh needs its neighbors, quickly.
Tree-based libraries improve this locality by ordering the leaves along a space-filling curve, which keeps cells that are close in space close in memory.
p4est, for example, computes its parallel partition with the Morton curve ([p4est forest documentation](https://github.com/cburstedde/p4est/blob/master/doc/doxygen/forest.dox)).

{{ project }} takes another route.
It never looks for the neighbors of one cell at a time: it describes whole groups of cells, such as "every cell of level $l$ whose right neighbor is at level $l + 1$", and walks through each group in a single ordered pass.
The rest of this page follows from that choice.

## Principles

### Store intervals, not cells

On a Cartesian grid, the cells of one level that sit next to each other along the x-direction form a run of consecutive integer indices.
{{ project }} stores such a run as one half-open interval $[a, b)$ instead of storing $b - a$ cells.
In 2D and 3D the same idea applies one direction up: the rows that hold x-intervals are themselves grouped into intervals along y, and along z in 3D, with offsets that point to the intervals of the next direction.
A `samurai::LevelCellArray` holds one level in this form, and a `samurai::CellArray` holds one per level.

```{figure} figures/philosophy_intervals.svg
:alt: A 1D mesh with eight cells at level 4 on the left half and two cells at level 2 on the right half. Stored cell by cell it needs ten entries; stored as intervals it needs two, [0, 8) at level 4 and [2, 4) at level 2.
:width: 80%
:align: center

The same 1D mesh stored cell by cell (10 entries) and as intervals (2 entries, one per level).
```

What this gives you:

- Memory follows the shape of the mesh, not its size.
  The mesh structure grows with the number of intervals, that is with the number of places where a run of cells starts or stops, and not with the number of cells.
  A large region refined at a single level costs a handful of intervals.
- Field values are contiguous along x.
  Each interval carries an `index` such that the value of cell $i$ is stored at position `index + i`.
  The values of an interval are a contiguous slice of the field storage, so a scheme applied to an interval works on whole arrays.
- Locality comes from the grid itself.
  The neighbor of cell $i$ along x is cell $i + 1$ of the same level, and the neighbor along y is found by translating the whole set by one row (see the next principle).
  No space-filling curve is needed to make neighbors cheap to reach.

What it costs:

- Looking up one isolated cell is not the fast path.
  Finding the interval that contains a given point scans the intervals of the matching row, one direction after the other (`samurai::find` in `algorithm.hpp`).
  Code that needs cells one at a time, in random order, pays for each lookup; code written as set operations does not.
- The compression depends on the mesh.
  A mesh where the level changes every cell has about as many intervals as cells, and the gain is lost.
- A level is immutable.
  A level cell array is sorted and frozen.
  To change the mesh, {{ project }} builds a new one from a `samurai::CellList`, the mutable counterpart, and transfers the fields onto it.
  Adaptation rebuilds the mesh instead of editing it in place.

The demo [`demos/p4est/simple_2d.cpp`](https://github.com/hpc-maths/samurai/blob/main/demos/p4est/simple_2d.cpp) rebuilds the mesh of the p4est `simple2` example and prints the memory used by each part of the mesh structure (`samurai::memory_usage` in `memory.hpp`).
For the real cells, it reports:

| Maximum level | Real cells | Mesh structure for the real cells | Per cell |
| --- | --- | --- | --- |
| 9 (default) | 66,379 | 33,688 bytes | 0.51 bytes |
| 12 (`--max-level 12`) | 4,200,559 | 263,616 bytes | 0.063 bytes |

These are counts, not timings: they do not depend on the machine.
They do depend on the mesh: in this example the band $x < 0.25$ is refined uniformly to the maximum level, which is the favorable case for intervals.
The values of a field come on top: a scalar field of `double` takes 8 bytes per cell whatever the mesh structure.

### Express mesh operations as set algebra

Once levels are sets of intervals, the questions an adaptive method asks become set expressions.
{{ project }} provides `intersection`, `union_`, `difference`, `translate`, `expand` and `contract`, and `.on(level)` brings an expression to a common level so that cells of different levels can be compared.
An expression holds no cells: it is evaluated lazily, and cells are produced only when it is traversed.
Because every operand is sorted, a traversal merges its operands in one ordered pass, the way two sorted lists are merged.

{{ project }} builds its own machinery from these expressions:

- the cells a scheme may read are the real cells grown by the stencil radius in every direction;
- the cells of level $l$ that receive the field projection of level $l + 1$ are the intersection of the two levels, brought to level $l$;
- with MPI, the cells shared with another rank are the intersection of that rank's cells and the local subdomain, both expanded by the stencil radius.

A subset operator, the kernel you pass to a set expression, then runs on every interval of the result.
The same mechanism serves the library and the user: the field projection in the README of the repository takes one set expression and four lines of arithmetic.
The {doc}`subset reference <reference/subset>` lists the operations, and the {doc}`subset tutorial <tutorial/operator_on_subset>` builds one step by step.

What it costs: you have to think in sets.
A loop over cells that tests the level of each neighbor is the natural first idea, and it is the slow one here.
Writing the same logic as an intersection or a difference is shorter once it is familiar, but it is a different habit.

### Decouple the mesh from the scheme

A scheme in {{ project }} is written for one cell, or for the flux across the interface between two cells of the same level.
It never handles a level jump itself: where a jump occurs, ghosts at the right level take part in the computation, so the flux is always computed between two cells of the same length.
Boundary conditions are attached to the field and fill the ghosts outside the domain.
The {doc}`finite volume schemes reference <reference/finite_volume_schemes>` describes the scheme side and the {doc}`boundary conditions reference <reference/bc>` the boundary side.

In practice, a scheme written once runs on both the multiresolution and the AMR meshes.
The level-set demos show it: [`level_set_MRA.cpp`](https://github.com/hpc-maths/samurai/blob/main/demos/FiniteVolume/level_set_MRA.cpp) and [`level_set_AMR.cpp`](https://github.com/hpc-maths/samurai/blob/main/demos/FiniteVolume/level_set_AMR.cpp) share the same scheme header, `level_set_schemes.hpp`, on a multiresolution mesh and on an AMR mesh.
Implementing a new scheme means writing its formula, not the mesh bookkeeping around it.

What it costs: the ghosts.
A mesh keeps, at every level, the ghost cells its schemes and its prediction may read, and filling them (field projection, prediction, boundary conditions, MPI exchange) takes time and memory at every step.
The mesh has to know in advance how far the schemes reach, which is why it is configured with a maximum stencil radius.

### One structure for multiresolution and AMR

The multiresolution mesh (`samurai::MRMesh`) and the AMR mesh (`samurai::amr::Mesh`) derive from the same base class, `samurai::Mesh_base`, and store the same thing: one cell array per mesh id (real cells, cells and ghosts, projection ghosts, and so on).
They differ in which collections they build and in how adaptation decides where to refine.
The fields, the set algebra, the schemes and the output functions are shared.

This is what lets {{ project }} offer both strategies without two code bases, and what lets a scheme move from one to the other, as the level-set demos above do.

## Static and dynamic set algebra

The set algebra exists in two forms.

### Static: the expression is a type

The static form, in `include/samurai/subset/`, encodes the whole expression in the C++ type system.
`intersection(a, translate(b, d)).on(level)` is a type, and the compiler sees the full tree: the traversal is inlined, with no virtual call and no heap allocation.
This is the form the library uses internally and the one to use in hot loops.
Its limit is that the structure of the expression must be known when the program is compiled.

### Dynamic: the expression is built at run time

The dynamic form, in `include/samurai/subset/dynamic/` under the `samurai::dyn` namespace, builds the same expressions at run time.
It covers the cases the static form cannot: an expression whose operands or operators are chosen at run time, and bindings to other languages such as Python, which cannot instantiate arbitrary C++ templates.
It reuses the static algorithms and erases the types at a single point, the traverser, behind a virtual interface; the rationale is in [pull request #485](https://github.com/hpc-maths/samurai/pull/485).

The price is speed.
The benchmark `benchmark/benchmark_dynamic_set.cpp` measures each pattern in both forms, building the expression inside the timed loop as a time loop does.
On an Apple silicon machine running Linux (aarch64), with GCC 15.3, a Release build and one thread, the dynamic form was 7 to 27 times slower than the static form across its eleven patterns: 12 to 18 times on three fixed boxes, 7 to 27 times on 2D and 3D multiresolution-adapted meshes.
Use the dynamic form when the structure is only known at run time; keep the static form everywhere else.

## Multiresolution and AMR: two adaptation strategies

Both strategies adapt the same kind of mesh.
They differ in how they decide where cells go.

### Multiresolution: driven by details

Multiresolution (MR) compares, at every cell, the field value with the value predicted from the level above.
This difference is the detail; a cell is coarsened where its detail is below a threshold, epsilon, scaled across levels by the assumed regularity of the solution.
The criterion does not depend on the equation being solved, and the threshold controls the error that adaptation adds.
Bellotti, Gouarin, Graille and Massot implement this approach for lattice Boltzmann methods in {{ project }}, with error control, and describe it as less problem-dependent than AMR approaches ([J. Comput. Phys. 2022, doi:10.1016/j.jcp.2022.111670](https://doi.org/10.1016/j.jcp.2022.111670); see also [arXiv:2102.12163](https://arxiv.org/abs/2102.12163) for the error analysis).
The cost: details need values at every level, so the field is projected onto the whole hierarchy, and one call to the adaptation runs up to one pass per level between the minimum and maximum levels (`samurai::make_MRAdapt`, see the {doc}`adaptation how-to <howto/adapt>`).

### AMR: driven by your criterion

AMR in {{ project }} lets you write the criterion yourself, evaluated on the field: a gradient, a level-set band, any rule that sets a tag per cell.
You then call `samurai::graduation` to keep neighboring cells within one level of each other and `samurai::update_field` to rebuild the mesh and transfer the field; [`demos/FiniteVolume/AMR_Burgers_Hat.cpp`](https://github.com/hpc-maths/samurai/blob/main/demos/FiniteVolume/AMR_Burgers_Hat.cpp) shows the loop.
The criterion can be cheap and tailored to the physics, but it is yours to design and tune for each problem, and {{ project }} gives no bound on the error it introduces.

### Choosing between them

Choose MR when you want an adaptation criterion that works across problems with a known error threshold; choose AMR when you already know which feature of the solution must be resolved.
The {doc}`graduation tutorial <tutorial/graduation>` explains the constraint both strategies rely on.

## Distributed parallel computing

The interval structure was designed with distributed computing in mind.
Under MPI, each rank owns a subdomain, and the cells it must exchange with another rank are computed with the same set algebra as everything else, by intersecting expanded sets (see the MPI case in the list above).
After adaptation the work drifts between ranks, and a load balancing module moves cells and fields to restore the balance; one of its strategies orders cells along a Hilbert space-filling curve.
Here the curve serves the partition, not the neighbor search.
See the {doc}`load balancing reference <reference/load_balancing>`.

```{admonition} TODO
:class: warning

TODO: The first design objective of {{ project }} is mesh handling more efficient than the combination of a tree and a space-filling curve.
This page has no benchmark that compares {{ project }} with a tree-based library such as p4est on the same problem, nor a parallel scaling study.
Add one, with its hardware, problem size, versions and timing method, or cite a publication that contains one.
```

## What {{ project }} does not aim to do

{{ project }} handles cell-based adaptation on Cartesian grids, in 1D, 2D and 3D, where every cell is a segment, a square or a cube at some level.
Several related needs are out of its scope, and other libraries serve them.

- Block-structured (patch-based) AMR.
  {{ project }} refines cell by cell; the patch-based and block-based variants are not implemented (see the feature list in the README of the repository).
  AMReX is "a software framework designed for building massively parallel block-structured adaptive mesh refinement (AMR) applications" ([AMReX introduction](https://amrex-codes.github.io/amrex/docs_html/Introduction.html)).
  Its documentation also lists support for particles, embedded boundary (cut cell) representations of complex geometries, and GPU acceleration with CUDA, HIP or SYCL backends ([AMReX README](https://github.com/AMReX-Codes/amrex#features)); {{ project }} provides none of these.
- Adaptive meshes on non-Cartesian geometry.
  A {{ project }} domain is a union of boxes, possibly with holes, on one Cartesian grid.
  p4est manages "a collection (a forest) of multiple connected adaptive quadtrees or octrees in parallel", whose roots form a conforming macro-mesh of quadrilaterals or hexahedra that covers the domain, and its documentation states that its algorithms "scale well to large numbers (10^6) of MPI processes" ([p4est README](https://github.com/cburstedde/p4est/blob/master/README)).
  It also provides the degree-of-freedom numbering that finite element codes need, and deal.II uses it to parallelize its mesh handling (same source).
- Unstructured meshes: triangles, tetrahedra and general polyhedra are outside the data structure.

## Related pages

- {doc}`tutorial/interval`: how a mesh is stored as intervals, level by level.
- {doc}`tutorial/operator_on_subset`: set expressions and subset operators in practice.
- {doc}`reference/subset`: the set algebra.
- {doc}`reference/finite_volume_schemes` and {doc}`reference/bc`: writing schemes and boundary conditions independently of the mesh.
- {doc}`howto/adapt`: multiresolution adaptation.
