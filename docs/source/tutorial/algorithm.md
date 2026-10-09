# Choosing between cell loops and interval loops

{{ project }} has two loops to visit the cells of a mesh: {cpp:func}`samurai::for_each_cell` calls a function once per cell, and {cpp:func}`samurai::for_each_interval` calls it once per interval of cells.
In this tutorial, we use both on a 1D field: we set the initial value of a field cell by cell, then we apply an upwind scheme to it interval by interval.
Along the way, we see which loop fits which kind of work.

This page covers the two loops called on all the cells of a mesh.
The {doc}`loop how-to <../howto/loop>` shows how to restrict them to ghost cells, one level or a subset, and how to run the cell loop in parallel.
The {doc}`algorithm API page <../api/algorithm>` lists every overload.

## Before you start

We assume you have read the {doc}`interval tutorial <interval>` and the {doc}`field tutorial <field>`: you know that {{ project }} stores the cells of each level as intervals along $x$, and that a field can be accessed with a cell, `u[cell]`, or with a level and an interval, `u(level, i)`.

The code of this tutorial comes from the first steps of the {doc}`1D Burgers tutorial <1d_burgers_amr>`, in `demos/tutorial/AMR_1D_Burgers/` of the {{ project }} repository.
To run it, you need a {{ project }} build directory, as described in the {doc}`installation how-to <../howto/installation>`.

## The mesh

Both steps use the same mesh: the segment $[-3, 3]$, with all its cells at level 6.
It is a {cpp:class}`samurai::CellArray` built from a box:

```{literalinclude} ../../../demos/tutorial/AMR_1D_Burgers/step_1/main.cpp
  :language: c++
  :start-at: constexpr std::size_t dim
  :end-at: mesh[init_level]
  :dedent:
```

All the cells are at the same level and touch each other, so this mesh holds a single interval that covers the whole segment.

## Setting the initial value cell by cell

We start with a hat function: $\phi(x) = 1 - |x|$ for $x \in [-1, 1]$ and $\phi(x) = 0$ elsewhere.
The function `init_sol` creates a scalar field `phi` on the mesh and sets its value in each cell:

```{literalinclude} ../../../demos/tutorial/AMR_1D_Burgers/step_1/init_sol.hpp
  :language: c++
  :start-at: template <class Mesh>
```

{cpp:func}`samurai::for_each_cell` takes the mesh and a function with one parameter, the cell.
It calls the function once for each cell of the mesh, at every level.
Inside the function:

- `cell.center(0)` is the $x$ coordinate of the cell center, a `double`.
- `phi[cell]` is the value of `phi` in this cell. We write to it here; we can read it the same way.

The function body is ordinary scalar C++: it handles one number at a time and can branch with `if` on the value of `x`.
This is what the cell loop is for: the value of each cell depends only on that cell, and the code reads best when it is written for one cell.

To see the result, build and run step 1 from the {{ project }} source directory:

```bash
cmake --build build --target tutorial-burgers1d-step-1
./build/demos/tutorial/AMR_1D_Burgers/step_1/tutorial-burgers1d-step-1
```

The program prints the mesh, then writes `amr_1d_burgers_step_1.h5` and `amr_1d_burgers_step_1.xdmf` in the current directory.
To plot the hat function from these files, follow the {doc}`plot how-to <../howto/plot>`.

## Applying a scheme interval by interval

Next, we solve the Burgers equation $\partial_t \phi + \partial_x (\phi^2 / 2) = 0$ with an upwind scheme.
The new value in cell $i$ depends on the value in cell $i$ and in its left neighbor $i - 1$:

$$
\phi_i^{n+1} = \phi_i^n - \frac{\Delta t}{2 \Delta x} \left( (\phi_i^n)^2 - (\phi_{i-1}^n)^2 \right)
$$ (tutorial-loop-upwind)

The function `update_sol` of step 2 computes {eq}`tutorial-loop-upwind` for all the cells:

```{literalinclude} ../../../demos/tutorial/AMR_1D_Burgers/step_2/update_sol.hpp
  :language: c++
  :start-at: template <class Field>
```

{cpp:func}`samurai::for_each_interval` takes the mesh and a function with three parameters:

- `level`: the level of the interval.
- `interval`: the interval, that is the $x$ indices of a run of consecutive cells.
- the indices in the other directions ($y$, $z$). The mesh is 1D, so this array is empty and the function leaves it unnamed.

Inside the function:

- `ii` is `interval` without its first and last cells.
- `phi(level, ii)` is an xtensor view on the values of `phi` for all the cells of `ii`.
- `ii - 1` is `ii` shifted by one cell to the left, so `phi(level, ii - 1)` holds the left neighbor of each cell of `ii`.
- The right-hand side is an xtensor expression over the whole interval. Assigning it to `phi_np1(level, ii)` computes the new values of all these cells in one statement.

On the mesh of this page, the loop calls the function once, on the interval $[0, 384)$ of level 6.
Its cell 0 starts at the left edge of the box, $x = -3$, and its cell 383 ends at the right edge, $x = 3$.
The diagram shows the two ends of this interval and of the intervals that the function builds from it:

```{diagram}
:figure: shifted_interval

Only the first and last cells of each row are drawn; the break mark replaces the cells 6 to 377.
Each arrow goes from a cell of `ii - 1` to its right neighbor in `ii`.
The bottom row is `interval - 1`: its first cell, $-1$, crossed in red, lies left of the box and is not in the mesh.
The dotted red lines mark the edges of the box, where the mesh ends.
```

The scheme reads a neighbor, and the interval loop expresses it directly: shifting the interval gives the values of all the neighbors at once, and the expression applies to the whole row of cells.
Since `phi(level, ii)` is a view and not a number, the math functions come from xtensor (`pow` here resolves to `xt::pow`), not from the C++ standard library.

The left neighbor of the first cell of the interval is outside the mesh, which is why `ii` leaves that cell out.
With `phi(level, interval - 1)` instead, the loop asks for the cell $-1$, which does not exist, and the program stops with a `std::out_of_range` exception that says the interval was not found.
The 1D Burgers tutorial replaces this trick with ghost cells in step 3.

Build and run step 2 the same way:

```bash
cmake --build build --target tutorial-burgers1d-step-2
./build/demos/tutorial/AMR_1D_Burgers/step_2/tutorial-burgers1d-step-2
```

The program prints one line `Iteration = ... Time = ...` per time step until the time reaches 1.5, then writes `amr_1d_burgers_step_2.h5` and `amr_1d_burgers_step_2.xdmf`.
For this initial condition, the exact solution forms a shock at $t = 1$, so the plot at $t = 1.5$ shows the hat moved to the right with a steep front.

## Which loop to use

Both loops visit the same cells; they differ in what the function receives.

| | {cpp:func}`samurai::for_each_cell` | {cpp:func}`samurai::for_each_interval` |
| --- | --- | --- |
| The function is called | once per cell | once per interval |
| It receives | a {cpp:class}`samurai::Cell`: level, center, length, indices | a level, an interval and the other indices |
| Field access | `u[cell]`, one value | `u(level, i)`, a view on the values of the interval |
| The body is written as | scalar code, with `if` and standard math functions | xtensor expressions, with xtensor math functions |
| Neighbors | not given by the cell | shifted intervals, `u(level, i - 1)` |

Use the cell loop when the value in a cell depends only on that cell, as for an initial condition or a pointwise source term: the code is written for one cell and can branch freely.
Use the interval loop when the computation reads neighboring cells, as for a stencil, or when you want to update a whole row of cells with one expression.

## What we did

We set a field cell by cell with {cpp:func}`samurai::for_each_cell`, then updated it interval by interval with {cpp:func}`samurai::for_each_interval`, reaching the neighbors of the cells by shifting the interval.

## Next steps

- {doc}`Loop how-to <../howto/loop>`: the same loops in 2D, on other cells than the solution cells, and in parallel.
- {doc}`1D Burgers tutorial <1d_burgers_amr>`: ghost cells and adaptive mesh refinement for the scheme of this page.
- {doc}`Operators on subsets <operator_on_subset>`: apply an operator on a part of the mesh built with the set algebra.
- {doc}`Algorithm API page <../api/algorithm>`: every overload of the loop functions.
