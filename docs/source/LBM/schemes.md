# Writing a lattice Boltzmann scheme

In this tutorial, we read and run a lattice Boltzmann program written with the `schemes/lbm` formalism of {{ project }}.
The program, `demos/LBM/new_D2Q4_advection.cpp`, solves a 2D advection equation with a D2Q4 scheme on a multiresolution mesh.
Along the way, we see how a scheme is described: a velocity set, a moment matrix, an equilibrium and relaxation parameters.
We then run the program, look at its output, and see how to add the lattice Boltzmann boundary conditions to a non-periodic case.

## Before you start

You need:

- a copy of the {{ project }} repository and a build directory configured with the demos (see the {doc}`installation how-to <../howto/installation>`);
- Python with `h5py` and `matplotlib` if you want to plot the result (see the {doc}`plot how-to guide <../howto/plot>`).

We assume you know the lattice Boltzmann method in its multiple-relaxation-time form.
The method on adapted meshes comes from {ref}`Bellotti et al. (2022a) <ref-lbm-mr-1d>` in 1D and {ref}`Bellotti et al. (2022b) <ref-lbm-mr-2d>` in 2D and 3D.
The {doc}`adaptation how-to <../howto/adapt>` explains the multiresolution calls that the program makes.

All the names used below are declared by `#include <samurai/schemes/lbm.hpp>`.

## The scheme in equations

A lattice Boltzmann scheme with $q$ velocities carries $q$ distribution functions $f_0, \dots, f_{q-1}$ in every cell.
Each $f_\alpha$ is attached to a lattice velocity $c_\alpha \in \mathbb{Z}^d$, where $d$ is the dimension.
In one time step $\Delta t$, $f_\alpha$ moves by $c_\alpha \Delta x$, where $\Delta x$ is the cell length at the finest level $L$ of the mesh.
The ratio $\lambda = \Delta x / \Delta t$ is the lattice velocity: the physical velocity attached to $f_\alpha$ is $\lambda c_\alpha$.

The collision works on the moments $m = (m_0, \dots, m_{q-1})$, obtained from the distributions by an invertible matrix $M$:

```{math}
:label: lbm-moments

m = M f,
\qquad
f = M^{-1} m.
```

The collision relaxes each moment towards its equilibrium value $m_k^{\mathrm{eq}}$, which is a function of the moments:

```{math}
:label: lbm-relaxation

m_k^\star = m_k + s_k \bigl( m_k^{\mathrm{eq}}(m) - m_k \bigr),
\qquad k = 0, \dots, q-1.
```

The relaxation parameter $s_k$ of a conserved moment is 0: the collision leaves that moment unchanged.
The other relaxation parameters usually lie in $(0, 2]$.
With $s_k = 1$, the collision sets $m_k$ to its equilibrium value.
The post-collision distributions are $f^\star = M^{-1} m^\star$.

On a uniform mesh at level $L$, the stream moves each post-collision distribution by one lattice velocity:

```{math}
:label: lbm-stream

f_\alpha(t + \Delta t, x) = f_\alpha^\star(t, x - c_\alpha \Delta x).
```

{{ project }} performs one time step as a stream followed by a collision, both applied to every cell of the adapted mesh.
The collision {eq}`lbm-relaxation` is local: each cell relaxes its own moments, whatever its level.
The stream on a coarser cell is described in [](#stream-on-the-adapted-mesh).

## The D2Q4 scheme for advection

The program approximates the solution $u(t, x, y)$ of

$$
\partial_t u + a_x \partial_x u + a_y \partial_y u = 0,
\qquad (x, y) \in [-1, 1]^2,
$$

with periodic boundaries, $a_x = a_y = 1/2$ and the initial condition $u(0, x, y) = \sin(\pi x) \sin(\pi y)$.
The exact solution is the initial condition shifted by $(a_x t, a_y t)$, which lets the program measure its error.

### Velocity set

The D2Q4 scheme has $q = 4$ velocities, which point to the four neighbors of a cell:

$$
c_0 = (1, 0), \quad c_1 = (0, 1), \quad c_2 = (-1, 0), \quad c_3 = (0, -1).
$$

```{diagram}
:figure: d2q4_velocities

The four velocities of the D2Q4 scheme, numbered as in the code.
```

### Moments

The four moments are the density $u$, two first-order moments and one second-order moment:

$$
m_0 = \sum_{\alpha} f_\alpha,
\qquad
m_1 = \lambda \sum_{\alpha} c_{\alpha, x} f_\alpha,
\qquad
m_2 = \lambda \sum_{\alpha} c_{\alpha, y} f_\alpha,
\qquad
m_3 = \lambda^2 \sum_{\alpha} \bigl( c_{\alpha, x}^2 - c_{\alpha, y}^2 \bigr) f_\alpha.
$$

Each row of $M$ lists the coefficients of one moment:

$$
M =
\begin{pmatrix}
1 & 1 & 1 & 1 \\
\lambda & 0 & -\lambda & 0 \\
0 & \lambda & 0 & -\lambda \\
\lambda^2 & -\lambda^2 & \lambda^2 & -\lambda^2
\end{pmatrix},
\qquad
M^{-1} =
\begin{pmatrix}
1/4 & 1/(2\lambda) & 0 & 1/(4\lambda^2) \\
1/4 & 0 & 1/(2\lambda) & -1/(4\lambda^2) \\
1/4 & -1/(2\lambda) & 0 & 1/(4\lambda^2) \\
1/4 & 0 & -1/(2\lambda) & -1/(4\lambda^2)
\end{pmatrix}.
$$

### Equilibrium and relaxation

The conserved moment is $m_0 = u$.
The first-order moments relax towards the fluxes of the advection equation, and the second-order moment towards zero:

$$
m_1^{\mathrm{eq}} = a_x u,
\qquad
m_2^{\mathrm{eq}} = a_y u,
\qquad
m_3^{\mathrm{eq}} = 0.
$$

The relaxation parameters are $s = (0, s_1, s_1, s_2)$, with $s_1 = 1.5$ and $s_2 = 1$ in the program.
With $s_1 < 2$, the scheme carries a numerical diffusion of order $\Delta t$, so it is first-order accurate; the run below measures that order.
The lattice velocity $\lambda = 1$ is larger than the advection velocities, as the scheme requires.

## The scheme in code

The program stores the distributions and the moments in two fields of the same mesh.

### Mesh and fields

The mesh is a periodic multiresolution mesh on $[-1, 1]^2$.
Without `--adapt`, its minimum and maximum levels are both `max_level`, so it is uniform.
`max_stencil_size(4)` sets the max stencil radius to 2:

```{literalinclude} ../../../demos/LBM/new_D2Q4_advection.cpp
  :language: c++
  :start-at: const Box box(
  :end-at: auto mesh
  :dedent:
```

The fields `f` (distributions) and `m` (moments) have one component per velocity.
The program sets the conserved moment $m_0$ to the initial condition and leaves the other moments at zero:

```{literalinclude} ../../../demos/LBM/new_D2Q4_advection.cpp
  :language: c++
  :start-at: auto m = samurai::make_vector_field
  :end-before: // D2Q4 scheme
  :dedent:
```

### Velocity scheme

The matrices are `std::array` objects, stored row by row.
Here is $M$; $M^{-1}$, named `invM`, is written the same way:

```{literalinclude} ../../../demos/LBM/new_D2Q4_advection.cpp
  :language: c++
  :start-at: const double l  = lambda;
  :end-before: std::array<std::array<double, 4>, 4> invM{
  :dedent:
```

The equilibrium is a function with two parameters.
The first one, `meq`, receives the $q$ equilibrium moments of the velocity scheme.
The second one, `mm`, holds the current moments of the cell.
The function must fill every entry of `meq`, but the scheme ignores the entries of the conserved moments.
The program then builds the scheme and sets the initial distributions:

```{literalinclude} ../../../demos/LBM/new_D2Q4_advection.cpp
  :language: c++
  :start-at: auto eq = [ax, ay]
  :end-at: scheme.init_equilibrium(f, m);
  :dedent:
```

`samurai::velocity_scheme<dim, q>` takes, in this order, the $q$ velocities $c_\alpha$, the matrices $M$ and $M^{-1}$, the relaxation parameters $s_k$ and the equilibrium.
It returns a `samurai::VelocityScheme<dim, q>`.

`samurai::make_lbm_scheme<field_t>` takes a name, the lattice velocity $\lambda$ and one or more velocity schemes, and returns a `samurai::LBMScheme`.
The sum of the $q$ of its velocity schemes must equal the number of components of `field_t`; the compiler checks it.
The name labels the timers printed with `--timers`.

`scheme.init_equilibrium(f, m)` sets the distributions to equilibrium: in each cell, it keeps the conserved moments of `m`, replaces the others by their equilibrium values, and applies $M^{-1}$.

### Time loop

The time step follows from the finest cell length, $\Delta t = \Delta x / \lambda$.
The adaptation is driven by the distributions `f`, which hold the whole state of the scheme:

```{literalinclude} ../../../demos/LBM/new_D2Q4_advection.cpp
  :language: c++
  :start-at: const double dx_fine
  :end-before: auto mass = [&]()
  :dedent:
```

Each iteration adapts the mesh, then calls the scheme once:

```{literalinclude} ../../../demos/LBM/new_D2Q4_advection.cpp
  :language: c++
  :start-at: for (std::size_t n = 0; n < nt; ++n)
  :end-before: std::cout << "case
  :dedent:
```

The adaptation transfers `f` to the new mesh, but not `m`: `m.resize()` only resizes `m` to the new mesh.
This is enough, because the scheme overwrites `m` in every cell.

`scheme(f, m)` performs one time step:

1. it fills the ghosts of `f`, which applies the boundary conditions attached to `f`;
2. it streams `f`;
3. it collides in every cell: $m = M f$, relaxation {eq}`lbm-relaxation`, $f = M^{-1} m^\star$.

```{diagram}
:figure: lbm_time_step

One call of `scheme(f, m)`. A source term, when the scheme has one, changes $m^\star$ between the relaxation and $f = M^{-1} m^\star$.
```

After the call, `m` holds the moments after relaxation.
Its conserved components are therefore the moments of the streamed distributions.
A third argument, the time step, is used only by a source term (see [](#several-velocity-schemes-and-source-terms)).

(stream-on-the-adapted-mesh)=

## Stream on the adapted mesh

At the finest level $L$, the stream is the shift {eq}`lbm-stream`: every value moves by a whole number of cells.
On a coarser cell, $c_\alpha \Delta x$ is only a fraction of the cell length.
During one time step, part of the content of the cell leaves it and part of the content of its upstream neighbors enters it.
{{ project }} measures these portions at level $L$, with the reconstructed stream of {ref}`Bellotti et al. (2022a) <ref-lbm-mr-1d>`.
Take a cell $C$ at level $\ell < L$; it covers $2^{d j}$ cells of level $L$, with $j = L - \ell$.

1. It predicts the post-collision distribution $f^\star_\alpha$ from level $\ell$ down to level $L$, on the cells of level $L$ inside $C$ and on those from which the stream reaches $C$. We write $\hat{f}^\star_\alpha$ for these predicted values.
2. It streams them at level $L$: each predicted value moves by $c_\alpha$, as on a uniform mesh at level $L$.
3. It averages: the new value of $C$ is the mean of the values that land in its cells of level $L$:

```{math}
:label: lbm-stream-mr

f_\alpha(t + \Delta t, C)
= \frac{1}{2^{d j}} \sum_{k \subset C} \hat{f}_\alpha^\star(t, k - c_\alpha),
```

where the sum runs over the cells $k$ of level $L$ inside $C$.
{ref}`plate-lbm-stream` follows these three steps for $f_0$ and $j = 1$, where $C$ covers $2 \times 2$ fine cells.

```{plate} The stream of a coarse cell
:figure: stream_coarse_cell
:label: plate-lbm-stream

**Fig. 1.** *coarse values*: $W$, $C$ and $E$ are cells of level $L - 1$, each with one value of $f^\star_0$.

**Fig. 2.** *predict*: the prediction gives four values $\hat{f}^\star_0$ to each cell, one per fine cell of level $L$.
Hatched in red, the fine cells whose value enters $C$ in the stream; shaded, those whose value stays in $C$; ruled, those whose value leaves $C$.

**Fig. 3.** *stream*: every value moves one fine cell to the east, by $c_0 = (1, 0)$.

**Fig. 4.** *average*: the new value of $C$ is the mean of the four values in its fine cells.
Only the hatched and the ruled columns change it:

$$
f_0(t + \Delta t, C) = f^\star_0(t, C) + \tfrac{1}{4} (h - r),
$$

where $h$ is the sum of the two hatched values and $r$ the sum of the two ruled values.

*The figures show one row of cells. The rows above and below enter the prediction of each fine value, but their contributions cancel in the mean of Fig. 4.*
```

The prediction keeps the mean: the average of $\hat{f}^\star_\alpha$ over the fine cells of $C$ is $f^\star_\alpha(t, C)$.
The values that move from one fine cell of $C$ to another, shaded in the plate, leave this average unchanged.
Equation {eq}`lbm-stream-mr` is therefore a balance between what enters $C$ and what leaves it:

```{math}
:label: lbm-stream-balance

f_\alpha(t + \Delta t, C)
= f^\star_\alpha(t, C)
+ \frac{1}{2^{d j}} \Bigl(
  \sum_{k \in I_\alpha} \hat{f}_\alpha^\star(t, k)
  - \sum_{k \in O_\alpha} \hat{f}_\alpha^\star(t, k)
\Bigr),
```

where $O_\alpha$ holds the fine cells of $C$ whose value leaves $C$ ($k \subset C$, $k + c_\alpha \not\subset C$), and $I_\alpha$ the fine cells outside $C$ whose value enters it ($k \not\subset C$, $k + c_\alpha \subset C$).
For $f_0$ in {ref}`plate-lbm-stream`, $O_0$ is the east column of $C$, ruled, and $I_0$ the east column of $W$, hatched in red.
Only these two portions change the value of $C$.
Both lie along the edges of $C$ that $c_\alpha$ crosses.

The code does not loop over the fine cells.
The prediction is linear, so the right-hand side of {eq}`lbm-stream-mr` is a fixed linear combination of the values of $f_\alpha$ around $C$ at level $\ell$.
Its coefficients depend only on $j$ and $c_\alpha$.
The scheme computes them once per level and per velocity, and applies them to every cell of the level.
For $c_0 = (1, 0)$, the cells they weigh lie on the row of $C$. The diagram labels each one by its offset from $C$, which is hatched:

```{diagram}
:figure: stream_stencil

The weights of the stream of $f_0$ for a cell $j = 1$ and $j = 2$ levels above the finest level.
```

With $j = 1$, the new value of $C$ is $\frac{9}{16} (f^\star_W + f^\star_C) - \frac{1}{16} (f^\star_{WW} + f^\star_E)$, where $WW$ is the west neighbor of $W$: half of $W$ and half of $C$, as in {ref}`plate-lbm-stream`, corrected by the prediction.
The weights add up to 1, so the stream keeps a constant field constant.
The same formula covers axis-aligned velocities, diagonal velocities and velocities longer than one cell.
For $j = 0$, it reduces to the shift {eq}`lbm-stream`.
For $c_\alpha = 0$, it leaves the value of the cell unchanged.

## Building and running the program

From the root of the repository, configure with the demos, build the target `lbm-new-D2Q4-advection`, and run it:

```bash
cmake -S . -B build -DBUILD_DEMOS=ON
cmake --build build --target lbm-new-D2Q4-advection
./build/demos/LBM/lbm-new-D2Q4-advection
```

The program runs to $t = 0.4$ on a uniform mesh at level 7 and prints:

```text
case = D2Q4 advection, uniform, max_level = 7, cells = 17424, dt = 0.015625, nt = 26, Tf_eff = 0.40625
mass drift = 5.44716e-18, u in [-0.993946, 0.993946]
relative L2 error = 0.00771854
```

The program rounds the number of time steps, so it stops at `Tf_eff`, the multiple of `dt` closest to the final time.
The `mass drift` is the change of $\int u$ over the run, at round-off level: the collision conserves $m_0$ and the stream on a periodic mesh moves it without loss.
The `relative L2 error` compares $u$ with the exact solution.
`cells` is the number of cells the mesh stores, ghosts included: the real cells are the $128^2 = 16384$ cells of level 7.
The last digits of the output can differ on another machine or compiler.

To check the order of the scheme, run it on several levels with the same final time, for example `--level 6 --Tf 0.5`.
With `--Tf 0.5`, the relative L2 error is 0.0410857 at level 5, 0.0194091 at level 6, 0.00940152 at level 7 and 0.00462286 at level 8.
It halves when the level increases by one: the scheme is first-order accurate.

### On an adapted mesh

Add `--adapt` to adapt the mesh between levels `--min-lvl` (2 by default) and `--level`, with the threshold `--eps` ($10^{-4}$ by default).
With the default values, the sine is smooth enough for the adaptation to coarsen the whole mesh to level 6, which shows little.
A larger threshold and a finer maximum level give a mesh with two levels:

```bash
./build/demos/LBM/lbm-new-D2Q4-advection \
    --adapt --level 8 --min-lvl 3 --eps 1e-2
```

```text
case = D2Q4 advection, adaptive, max_level = 8, min_level = 3, cells = 5288, dt = 0.0078125, nt = 51, Tf_eff = 0.398438
mass drift = 3.69984e-18, u in [-0.990061, 0.990061]
relative L2 error = 0.0063968
```

The time step is set by level 8, so the run takes 51 steps.
At the end, the mesh has 1756 real cells: 780 at level 5 and 976 at level 6.
The uniform mesh at level 8 has $256^2 = 65536$ cells; with the same final time, its relative L2 error is 0.00370543 (run with `--level 8`).
The adapted mesh uses 37 times fewer cells, for an error 1.7 times larger.

The program writes the moments to `new_D2Q4_advection.h5` and `new_D2Q4_advection.xdmf`, in the directory given by `--path` (the current directory by default); `--filename` changes the file name.
The cells at level 6 gather where the gradient of $u$ is largest, half-way between neighboring extrema:

```{figure} ./figures/D2Q4_advection.png
:width: 100%
:align: center
:alt: Two square plots of the domain [-1, 1] squared. On the left, the cell levels: a background of level 5 cells with eight clusters of level 6 cells, each one half-way between two neighboring bumps of the solution, plus a few isolated level 6 cells. On the right, the solution: two red and two blue bumps in a checkerboard, centered near (-0.3, -0.3), (0.7, 0.7), (-0.3, 0.7) and (0.7, -0.3).

Run with `--adapt --level 8 --min-lvl 3 --eps 1e-2`, at $t = 0.398438$: levels of the cells (left) and $u = m_0$ (right).
```

## Boundary conditions

The D2Q4 advection program uses a periodic mesh, so it attaches no boundary condition.
On a non-periodic mesh, attach the conditions to the distribution field `f` with `samurai::make_bc`, before calling `init_equilibrium` and adapting the mesh.
The scheme applies them when it fills the ghosts of `f`, before the stream reads these ghosts.
The {doc}`boundary conditions reference <../reference/bc>` describes how conditions attach to a field and which boundary they act on.

### Walls: bounce-back and anti-bounce-back

`samurai::BounceBack` and `samurai::AntiBounceBack` write in the ghost a reflection of the inner cell next to it.
Let $\bar\alpha$ be the index of the opposite velocity, $c_{\bar\alpha} = -c_\alpha$.
The ghost receives

$$
f_\alpha^{\mathrm{ghost}} = \sigma f^{\mathrm{inner}}_{\bar\alpha} + r_\alpha,
\qquad
r_\alpha = f^{\mathrm{eq}}_\alpha(m_w) - \sigma f^{\mathrm{eq}}_{\bar\alpha}(m_w),
$$

with $\sigma = 1$ for bounce-back and $\sigma = -1$ for anti-bounce-back.
After the stream, the distribution that enters the inner cell from the ghost is the reflection of the one that left it.
Bounce-back imposes the odd moments, such as the momentum, and gives a wall.
Anti-bounce-back imposes the even moments, such as the density or the water height, and gives a Dirichlet condition on them.
The term $r_\alpha$ carries the imposed value through an equilibrium distribution $f^{\mathrm{eq}}(m_w)$; it is zero when you pass no equilibrium.
For a velocity $c_\alpha$ without opposite in the set, such as the rest velocity, $\bar\alpha = \alpha$.

At the left wall of the dam break below, with the D1Q3 velocities $c_0 = 0$, $c_1 = 1$ and $c_2 = -1$, the ghost turns the $f_2$ that leaves the inner cell into the $f_1$ that comes back.
This wall is a bounce-back without equilibrium, so $\sigma = 1$, $r_1 = 0$ and the ghost receives $f_1 = f_2^\star$:

```{diagram}
:figure: bounce_back

The bounce-back of $f_2$ at a wall, over one time step.
The wall is the face between the ghost (dashed) and the inner cell; the reflection is in red.
```

The shallow water dam break `demos/LBM/new_D1Q3_shallow_waters_dam.cpp` (target `lbm-new-D1Q3-shallow-waters-dam`) uses both.
It lists the lattice velocities once more, for the boundary conditions:

```{literalinclude} ../../../demos/LBM/new_D1Q3_shallow_waters_dam.cpp
  :language: c++
  :start-at: const std::array<std::array<int, dim>, 3> velocities{
  :end-at: "};"
  :dedent:
```

With `--bc antibounceback`, the left side is a wall and the right side is a reservoir at the height $h_R$ (option `--hR`):

```{literalinclude} ../../../demos/LBM/new_D1Q3_shallow_waters_dam.cpp
  :language: c++
  :start-at: auto reservoir = [&scheme]
  :end-at: ->on(right);
  :dedent:
```

Without this option, `samurai::make_bc<samurai::BounceBack>(f, velocities)` puts a wall on both sides.

The `velocities` argument is the list passed to `samurai::velocity_scheme`; the condition uses it to find the opposite velocities.
The third argument of `make_bc` is $f^{\mathrm{eq}}(m_w)$: either a fixed distribution, or, as here, a function that computes it from the distributions of the inner cell.
`scheme.moments` and `scheme.equilibrium_f` give the moments of a distribution and the equilibrium distribution of a moment vector.

When the velocity set contains diagonal velocities (D2Q9, D2Q4 with diagonal velocities), the reflection is also applied in the corner ghosts of the domain.

### Inflow and outflow

`samurai::ImposedDistribution` writes a fixed distribution in the ghost, so that the stream brings it into the domain: an inflow.
The usual distribution is the equilibrium of the incoming flow.
On the outflow side, the finite volume condition `samurai::Neumann<1>` copies the inner distributions into the ghost.
The von Kármán street `demos/LBM/new_D2Q9_von_karman.cpp` (target `lbm-new-D2Q9-von-karman`) sets an inflow on three sides and an outflow on the right:

```{literalinclude} ../../../demos/LBM/new_D2Q9_von_karman.cpp
  :language: c++
  :start-at: const std::array<double, 9> feq_inflow
  :end-at: make_bc<samurai::Neumann<1>>(f)->on(right);
  :dedent:
```

`samurai::ImposedDistribution` does not fill the corner ghosts: they keep the value extrapolated from the inner cells.

(several-velocity-schemes-and-source-terms)=

## Several velocity schemes and source terms

`samurai::make_lbm_scheme` accepts several velocity schemes, for the vectorial schemes of systems such as the Euler equations.
Their moments are concatenated in one field, in the order of the arguments.
The `mm` argument of every equilibrium holds this whole moment vector, so the equilibrium of one velocity scheme can depend on the moments of the others.
The D1Q222 scheme for the Sod problem, in `demos/LBM/new_D1Q222_euler_sod.cpp`, uses three D1Q2 velocity schemes.
Each one has the velocities $+1$ and $-1$ and carries one conserved variable: the density $\rho$, the momentum $q = \rho u$ and the total energy $E$.

```{diagram}
:figure: d1q222_layout

The six components of `f` and `m` in the D1Q222 scheme.
The arrows above `f` give the velocity of each component, $+1$ or $-1$.
Each velocity scheme maps its two components of `f` to its two moments with its own $M$.
The conserved moments, with $s_k = 0$, are hatched in red.
```

With several velocity schemes, a wall that lets the flow slip along it reverses only the normal momentum.
The overload `samurai::make_bc<samurai::BounceBack>(f, velocities, block_sizes, block_odd_axis)` takes the $q$ of each velocity scheme and, for each one, the axis of the momentum it carries, or `-1` for a scalar such as the density or the energy.
`demos/LBM/new_D2Q4444_euler_implosion.cpp` uses it.

`scheme.set_source(source)` registers a source term, such as gravity.
The function `source(m, dt)` receives the moment vector of a cell, after relaxation, and may modify it before $f = M^{-1} m^\star$.
Pass the time step as the third argument of `scheme(f, m, dt)`.
`demos/LBM/new_D2Q5444_euler_rayleigh_taylor.cpp` uses it for gravity.

## Next steps

- The {doc}`test cases <test_cases>` page lists the other programs of `demos/LBM`, with their equations, parameters and reference results.
- The {doc}`adaptation how-to <../howto/adapt>` explains the threshold and the other options of the multiresolution adaptation.
- The API reference does not cover the lattice Boltzmann classes; their documentation is in the headers of `include/samurai/schemes/lbm/`.
