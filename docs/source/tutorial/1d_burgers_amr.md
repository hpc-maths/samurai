# A finite volume solver for the 1D Burgers equation

In this tutorial, we write a {{ project }} program that solves the 1D Burgers equation with a finite volume scheme on an adaptive mesh.
The initial condition is a hat function, and the simulation runs past the time at which a shock forms.
At each time step, a refinement criterion adapts the mesh to the solution: small cells where the solution varies, large cells where it is constant.
At the end, we have a program that writes the solution and the levels of the cells to an HDF5 file, and a plot to compare with the exact solution.

## Before you start

You need:

- {{ project }} installed (see the {doc}`installation guide <../howto/installation>`);
- a CMake project that builds a program against {{ project }} (see the {doc}`CMake how-to guide <../howto/cmake>`): we write the code of this tutorial in its `main.cpp` file;
- Python with `h5py` and `matplotlib`, and a copy of the {{ project }} repository, to plot the result (see the {doc}`plot how-to guide <../howto/plot>`).

The tutorial assumes you know the basics of finite volume methods for conservation laws.
The {doc}`interval <interval>` and {doc}`field <field>` tutorials explain how {{ project }} stores the mesh and the fields that we use here.

## The problem

We approximate the solution $u: [0, T] \times \mathbb{R} \to \mathbb{R}$ of the 1D Burgers equation

```{math}
:label: burgers-equation

\begin{cases}
    \partial_t u(t, x) + \partial_x \varphi(u(t, x)) = 0, & t \in [0, T], \quad x \in \mathbb{R}, \\
    u(0, x) = u_0(x), & x \in \mathbb{R},
\end{cases}
\qquad \text{with} \quad \varphi(u) = \frac{u^2}{2},
```

where $T > 0$ is the final time and $\varphi$ is the flux.
The initial condition is the hat function

```{math}
:label: burgers-initial-condition

u_0(x) = (1 + x) \chi_{[-1, 0]}(x) + (1 - x) \chi_{[0, 1]}(x),
```

where $\chi_I$ is the indicator function of the interval $I$: it is 1 on $I$ and 0 elsewhere.

### Exact solution

Since $\varphi'(u) = u$, the characteristics are the straight lines $x = x_0 + u_0(x_0) t$.
Along each of them, $u$ keeps the value $u_0(x_0)$.
The characteristics that start in $[0, 1]$ all meet at $x = 1$ at time

$$
T^\star = -\frac{1}{\min_{x} u_0'(x)} = 1.
$$

For $t < T^\star$, following the characteristics gives the exact solution

$$
u(t, x) = \frac{1 + x}{1 + t} \chi_{[-1, t]}(x) + \frac{1 - x}{1 - t} \chi_{[t, 1]}(x).
$$

Its slope on $[t, 1]$ is $-1/(1 - t)$, which tends to $-\infty$ as $t \to T^\star$: a shock forms at $x = 1$ at time $T^\star$.
For $t \geq T^\star$, the solution is

```{math}
:label: burgers-solution-after-shock

u(t, x) = \frac{1 + x}{1 + t} \chi_{[-1, s(t)]}(x), \qquad s(t) = \sqrt{2 (1 + t)} - 1.
```

The shock position $s(t)$ follows from the Rankine-Hugoniot condition $s'(t) = \big(u(t, s(t)^-) + 0\big) / 2$, with $s(T^\star) = 1$.
The solution therefore stays bounded, with $0 \leq u \leq 1$, but changes regularity at $T^\star$:

$$
\begin{cases}
    u(t, \cdot) \in C^0(\mathbb{R}) \cap L^\infty(\mathbb{R}), & t \in [0, T^\star), \\
    u(t, \cdot) \in L^\infty(\mathbb{R}), \ \text{discontinuous at } x = s(t), & t \in [T^\star, T].
\end{cases}
$$

We run the simulation up to $T = 1.5$.
At that time, the shock is at $s(1.5) = \sqrt{5} - 1 \approx 1.236$, and the solution just left of the shock is $2 / \sqrt{5} \approx 0.894$.

## The finite volume scheme

### Mesh

We work on the bounded domain $[a, b] = [-3, 3]$.
It contains the support of the solution up to $T = 1.5$, which is $[-1, s(1.5)]$, so the solution is zero near the boundaries.
Splitting $[a, b]$ into $2^j$ cells of equal size gives the cells of level $j$:

```{math}
:label: burgers-cells

C_{j,k} = [x_{j,k-1/2}, x_{j,k+1/2}], \qquad x_{j,k-1/2} = a + k \Delta x_j, \qquad \Delta x_j = \frac{b - a}{2^j}, \qquad k = 0, \dots, 2^j - 1,
```

with center $x_{j,k} = a + (k + 1/2) \Delta x_j$.
In {{ project }}, $j$ is the level of the cell and $k$ its index at that level.
Splitting $C_{j,k}$ in two gives the cells $C_{j+1,2k}$ and $C_{j+1,2k+1}$ of level $j + 1$.

An adaptive mesh is a set of cells of different levels that covers $[a, b]$ without overlap.
The levels lie between a minimum level $j_{\min}$ and a maximum level $J$.
We use $j_{\min} = 2$, cells of size $1.5$, and $J = 8$, cells of size $\Delta x_J = 6 / 256 = 0.0234375$.
The mesh is graded: two neighboring cells differ by at most one level (see the {doc}`graduation tutorial <graduation>`).

### Time stepping

Time is discretized with a time step $\Delta t$, so that $t^n = n \Delta t$ for $n = 0, 1, \dots$
To write the scheme, we number the cells of the mesh from left to right, $C_k$ with size $\Delta x_k$, whatever their level.
The unknowns are the cell averages

$$
\overline{u}_k^n \simeq \frac{1}{\Delta x_k} \int_{C_k} u(t^n, x) \, \text{d}x.
$$

The finite volume scheme reads

```{math}
:label: burgers-fv-scheme

\overline{u}^{n+1}_k = \overline{u}^n_k - \frac{\Delta t}{\Delta x_k} \left( F_{k+1/2}^n - F_{k-1/2}^n \right),
```

where $F_{k-1/2}^n = \mathcal{F}(\overline{u}^n_{k-1}, \overline{u}^n_k)$ is the numerical flux through the interface $x_{k-1/2}$ between $C_{k-1}$ and $C_k$.

### Upwind flux

We use the upwind flux, which takes the flux of the cell the information comes from:

```{math}
:label: burgers-upwind-flux

\mathcal{F}(\overline{u}_L, \overline{u}_R) =
\begin{cases}
    \varphi(\overline{u}_L), & \text{if} \quad \dfrac{\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R)}{2} \geq 0, \\
    \varphi(\overline{u}_R), & \text{if} \quad \dfrac{\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R)}{2} < 0.
\end{cases}
```

The solution of our problem stays non-negative, because $u_0 \geq 0$ (maximum principle).
Then $\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R) = \overline{u}_L + \overline{u}_R \geq 0$, and the upwind flux is always $\varphi(\overline{u}_L)$.
The scheme {eq}`burgers-fv-scheme` becomes

```{math}
:label: burgers-upwind-scheme

\overline{u}^{n+1}_k = \overline{u}^n_k - \frac{\Delta t}{2 \Delta x_k} \left( (\overline{u}^n_k)^2 - (\overline{u}^n_{k-1})^2 \right).
```

Another choice is the Lax-Friedrichs flux, which is more diffusive than the upwind flux:

$$
\mathcal{F}(\overline{u}_L, \overline{u}_R) = \frac{1}{2} \left( \varphi(\overline{u}_L) + \varphi(\overline{u}_R) \right) - \frac{\Delta x}{2 \Delta t} \left( \overline{u}_R - \overline{u}_L \right),
$$

written here for cells of equal size $\Delta x$.

### CFL condition

The time step must satisfy the CFL condition

```{math}
:label: burgers-cfl

\Delta t \leq \frac{\min_k \Delta x_k}{\sup_{x \in \mathbb{R}} |\varphi'(u_0(x))|}.
```

Since $\varphi'(u) = u$ and $0 \leq u \leq \sup u_0 = 1$ at all times, the condition is $\Delta t \leq \Delta x_J$, the size of the smallest cells the mesh can hold.
We take $\Delta t = 0.99 \, \Delta x_J$, that is a CFL number of 0.99, and keep it when the mesh changes.
With a CFL number above 1, the scheme is unstable: oscillations appear and grow.

(burgers-mesh-adaptation)=

### Mesh adaptation

The shock and the corners of the solution need small cells, while large cells are enough where the solution is constant.
Before each time step, we split the cells where the solution varies, and merge the cells where it does not, with the criterion

```{math}
:label: burgers-criterion

\text{split } C_{j,k} \quad \text{if} \quad |\partial_x \overline{u}_{j,k}| > \delta, \qquad \text{merge it otherwise},
```

where the threshold $\delta$ is a parameter and the derivative is estimated with the centered formula

$$
\partial_x \overline{u}_{j,k} \simeq \frac{\overline{u}_{j,k+1} - \overline{u}_{j,k-1}}{2 \Delta x_j}.
$$

Merging $C_{j,k}$ means replacing it and its sibling by their parent cell of level $j - 1$.
A cell is never split beyond level $J$ nor merged below level $j_{\min}$.
On the new mesh, a merged cell takes the mean of the values of its two children, so the integral of the solution does not change.
A split cell gives its value to its two children, corrected by the slope of the solution on its neighbors, so that their mean is still the value of the parent.
We take $\delta = 0.1$.

## Build the mesh

We start the `main` function by initializing {{ project }} and reading the command-line options:

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
:start-after: // Read the options
:end-before: // Build the mesh
:dedent:
```

{cpp:func}`samurai::initialize` returns the command-line application, to which we add the final time, the CFL number and the threshold $\delta$ of the criterion as options.
`always_capture_default` makes `--help` show the default value of each of them.
`SAMURAI_PARSE` reads them, together with the options of {{ project }}.

Next, we describe the mesh with {cpp:class}`samurai::mesh_config`, with its minimum and maximum levels, and build it on the box $[-3, 3]$ with `samurai::amr::make_mesh`:

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
:start-after: // Build the mesh
:end-before: // Set the initial condition
:dedent:
```

The mesh is an adaptive mesh refinement (AMR) mesh.
It starts uniform at the maximum level: the level-0 cell is the whole box, so a cell at level 8 has the size $\Delta x_8 = 6 / 2^8$.
The program prints the sub-meshes of the mesh (`cells`, `cells and ghosts`, ...).
The `cells` sub-mesh has a single interval `[0,256)` at level 8: the 256 cells $C_{8,0}, \dots, C_{8,255}$ of {eq}`burgers-cells`.
The first adaptation of the time loop merges the cells where the initial condition is constant.

:::{remark}
The command-line options `--min-level` and `--max-level` replace the levels set in the code (see the {doc}`mesh how-to guide <../howto/mesh>`).
With the same value for both, the mesh stays uniform.
:::

## Set the initial condition

We create a scalar field `u` on the mesh with {cpp:func}`samurai::make_scalar_field`, and set it to the hat function {eq}`burgers-initial-condition` at the center of each cell:

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
:start-after: // Set the initial condition
:end-before: // Set the boundary condition
:dedent:
```

Taking the value at the center instead of the cell average is accurate to second order in the cell size where $u_0$ is smooth.

The scheme {eq}`burgers-upwind-scheme` in the first cell $C_0$ reads $\overline{u}_{-1}$, which lies outside the domain.
{{ project }} stores such values in ghost cells, filled from the boundary conditions of the field.
We attach a homogeneous Neumann condition to `u`, which copies the value of the boundary cell into the ghost cells:

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
:start-after: // Set the boundary condition
:end-before: // Define the scheme
:dedent:
```

The solution is zero near $x = \pm 3$ up to $T = 1.5$, so this choice does not change the result.
The {doc}`boundary conditions reference <../reference/bc>` lists the other conditions.

## Define the scheme

{{ project }} provides the upwind flux {eq}`burgers-upwind-flux` as a ready-made operator, `samurai::make_convection_upwind`.
It computes the flux $u^2$, so we multiply it by $1/2$ to get $\varphi(u) = u^2/2$:

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
:start-after: // Define the scheme
:end-before: // Write the time loop
:dedent:
```

The operator is defined by its numerical fluxes at the cell faces.
Each face gets one flux, shared by the two cells on either side, which keeps the scheme conservative, also on the faces between cells of different levels.
`conv(u)` returns $(F_{k+1/2}^n - F_{k-1/2}^n) / \Delta x_k$ for every cell $C_k$, the term of {eq}`burgers-upwind-scheme` that multiplies $\Delta t$.
The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` describes this operator and the others.

## Define the refinement criterion

The criterion {eq}`burgers-criterion` fills a tag field, with one value per cell: `samurai::CellFlag::refine` to split the cell, `samurai::CellFlag::coarsen` to merge it, `samurai::CellFlag::keep` to leave it as it is.
We write it in a function, before `main`:

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
:start-after: // Define the refinement criterion
:end-before: int main
```

The lambda `tag_interval` handles one interval of cells, and `samurai::for_each_interval` calls it on the intervals of each level.
On an interval `i` of level `level`, `u(level, i + 1)` is the field on the interval shifted one cell to the right, that is $\overline{u}_{j,k+1}$ for all the cells $k$ of the interval at once, so `du_dx` holds the estimate of $|\partial_x \overline{u}_{j,k}|$ of every cell of the interval.
The neighbors of the cells at both ends of an interval are ghost cells: they lie outside the domain, or the mesh has no cell of the same level there.
`samurai::apply_on_masked` applies `refine` or `coarsen` to the tags of the cells where the condition holds.
The tests on `level` keep the levels between the minimum and the maximum level.

## Write the time loop

The scheme needs a second field for $\overline{u}^{n+1}$, a tag field for the criterion, and the time step from the CFL condition {eq}`burgers-cfl`.
`mesh.min_cell_length()` is the size $\Delta x_J$ of the cells of the maximum level.
Each iteration of the loop does four things:

1. it adapts the mesh to `u`, in passes; each pass
   - fills the ghost cells of `u` with `samurai::update_ghost`, because the criterion reads the neighbors of each cell;
   - resizes the tag field to the current mesh, and fills it with the criterion;
   - changes the tags with {cpp:func}`samurai::graduation` so that the new mesh is graded, along the directions of `stencil`, here left and right;
   - builds the new mesh from the tags with `samurai::update_field`, and moves `u` onto it as described in {ref}`burgers-mesh-adaptation`;

   `samurai::update_field` returns `true` when the tags leave the mesh unchanged, and the passes stop there.
   A pass changes a cell by one level at most, so `npasses`, the difference between the maximum and the minimum level, is enough to take a cell from one end of the level range to the other; the multiresolution adaptation of {{ project }} uses the same bound.
   The bound also stops the passes when a cell is split at one pass and merged back at the next, which happens in 9 of the 65 time steps of this run.
   After the passes, `unp1.resize()` sizes `unp1` to the new mesh: its values do not matter, since the scheme overwrites them;
2. it advances the time, and shortens the last time step so that the loop stops exactly at `Tf`;
3. it computes $\overline{u}^{n+1} = \overline{u}^n - \Delta t \, \mathrm{conv}(\overline{u}^n)$; `conv(u)` first fills the ghost cells of `u` it reads, if they are out of date, applying the boundary condition;
4. it exchanges the values of `u` and `unp1` with {cpp:func}`samurai::swap`, so that `u` holds the new solution for the next iteration.

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
:start-after: // Write the time loop
:end-before: // Save the solution
:dedent:
```

{cpp:func}`samurai::swap` exchanges the data of the two fields, and also whether their ghost cells are up to date.
The boundary condition stays attached to `u`.

After the loop, we store the level of each cell in a field `level`, and save the mesh, the solution and the levels to `burgers_1d.h5`, in the directory the program runs from:

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
:start-after: // Save the solution
:end-before: return 0;
:dedent:
```

## Run the program and plot the result

The complete program, with the pieces above in order, is compiled with the documentation snippets:

```{literalinclude} snippet/burgers/burgers_1d.cpp
:language: c++
```

1. Build the program as described in the {doc}`CMake how-to guide <../howto/cmake>`.
2. Run it.
   After the description of the mesh, it prints one line per time step.
   With the default options, $\Delta t = 0.99 \times 6 / 256 \approx 0.0232$, and the last line is:

   ```text
   iteration 65: t = 1.5, 115 cells
   ```

   The program writes `burgers_1d.h5` and `burgers_1d.xdmf` in the current directory.
3. Plot the solution with the script of the {{ project }} repository, where `<samurai-dir>` is the root of your copy:

   ```bash
   python <samurai-dir>/python/read_mesh.py burgers_1d --field u level
   ```

The plot of `u` shows the ramp $u = (1 + x) / 2.5$ of {eq}`burgers-solution-after-shock`, from $x = -1$ up to the shock near $x \approx 1.236$, where $u$ drops from about 0.89 to 0.
The upwind scheme spreads the shock and the corner at $x = -1$ over a few cells.

The plot of `level` shows the mesh.
The slope of the ramp, $0.4$, is above $\delta = 0.1$, so the cells are at level 8 from $x \approx -1.03$ to $x \approx 1.31$, past the shock.
Away from the ramp, the solution is zero, and the level drops by one every one or two cells, as fast as the graduation allows, down to level 3 near the boundaries.
The mesh has 115 cells instead of 256.

Change the options to see their effect:

- `--min-level 8` keeps the mesh uniform at level 8: the solution is almost the same, with 256 cells;
- `--max-level 10` sharpens the shock and the corner: $\Delta x_J$ is 4 times smaller, so there are 4 times more time steps, and the mesh ends with 407 cells instead of 1024;
- `--delta 0.5` merges the cells of the ramp, whose slope is below the threshold: the mesh ends with 29 cells, but the solution is less accurate on the ramp.

## What we built

We solved the Burgers equation {eq}`burgers-equation` past the formation of a shock.
On the way, we built an AMR mesh from a `samurai::mesh_config`, created and initialized a scalar field, attached a boundary condition, wrote a refinement criterion, and wrote a finite volume time loop that adapts the mesh before each time step and applies the flux-based upwind operator.

## Next steps

- The demos in [demos/tutorial/AMR_1D_Burgers](https://github.com/hpc-maths/samurai/tree/main/demos/tutorial/AMR_1D_Burgers) solve the same problem step by step, and write by hand the graduation, the mesh update and the ghost update that `samurai::graduation`, `samurai::update_field` and `samurai::update_ghost` do here.
- The {doc}`loop how-to guide <../howto/loop>` and the {doc}`field how-to guide <../howto/field>` cover more ways to loop over the mesh and to access fields.
- The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` describes the operators {{ project }} provides for finite volume schemes.
