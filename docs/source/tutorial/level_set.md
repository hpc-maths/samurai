# Level-set transport on an adaptive mesh

In this tutorial, we follow a circle that a vortex flow stretches into a spiral.
The circle is the zero level of a level-set function, which we transport with a finite volume scheme on an adaptive mesh refinement (AMR) mesh between levels 4 and 8.
At each time step, we adapt the mesh around the contour, transport the level-set function, then reinitialize it.

The code shown on this page comes from the demo [`demos/FiniteVolume/level_set_AMR.cpp`](https://github.com/hpc-maths/samurai/blob/main/demos/FiniteVolume/level_set_AMR.cpp) and its schemes in [`demos/FiniteVolume/level_set_schemes.hpp`](https://github.com/hpc-maths/samurai/blob/main/demos/FiniteVolume/level_set_schemes.hpp).

## Before you start

- You have built {{ project }} from source (see {ref}`build-from-source`): the demo is part of the source tree, not of the installed library.
- You know how to create a mesh and a field (see the {doc}`mesh how-to guide <../howto/mesh>` and the {doc}`field tutorial <field>`).
- To look at the results, you have ParaView (see the {doc}`plot how-to guide <../howto/plot>`).

## Run the demo

From the root of the source tree, with the build directory `build` you configured, build and run the demo:

```bash
cmake --build build --target finite-volume-level-set-amr
./build/demos/FiniteVolume/finite-volume-level-set-amr
```

The demo prints one line per time step, from `iteration 0: t = 0.00244140625, dt = 0.00244140625` to the final time $t = 3.14$, then the timers.
It writes the level set, the velocity and the level of each cell at the final time in `FV_level_set_2d_AMR.h5` and `FV_level_set_2d_AMR.xdmf`, in the current directory, and the mesh and the level set in the restart file `FV_level_set_2d_AMR_restart.h5`.
Open the `.xdmf` file in ParaView to see the mesh and the contour $\phi = 0$.

To save intermediate steps, add `--nfiles 10`: the demo then writes ten files, `FV_level_set_2d_AMR_ite_1.h5` to `FV_level_set_2d_AMR_ite_10.h5`.
`--help` lists the other options, among them `--Tf` (final time), `--cfl`, `--min-level` and `--max-level`.

## The level-set method

A contour $\Gamma(t)$ in the plane is the zero level of a scalar function $\phi(t, \mathbf{x})$, the level-set function:

$$
\Gamma(t) = \left\{ \mathbf{x} \in \mathbb{R}^2 \;:\; \phi(t, \mathbf{x}) = 0 \right\}.
$$ (eq-level-set-contour)

We take $\phi < 0$ inside the contour and $\phi > 0$ outside.
The outward normal vector $\mathbf{n}$ and the mean curvature $\kappa$ of the contour follow from derivatives of $\phi$ (see {ref}`Osher and Fedkiw <ref-osher-fedkiw>`, and {ref}`Gibou, Fedkiw and Osher <ref-gibou-2018>` for a review of level-set methods):

$$
\mathbf{n} = \dfrac{\nabla \phi}{\left| \nabla \phi \right|}, \qquad \kappa = \nabla \cdot \mathbf{n} = \nabla \cdot \left( \dfrac{\nabla \phi}{\left| \nabla \phi \right|} \right).
$$

The contour moves with a velocity field $\mathbf{u} = (u, v)$, divergence-free everywhere.
The level-set function then follows the passive transport equation (also called the color equation):

$$
\partial_t \phi + \mathbf{u} \cdot \nabla \phi = 0.
$$ (eq-level-set-transport)

A level-set function is easiest to use when it is a signed distance to the contour ($\left| \nabla \phi \right| = 1$).
Transport by {eq}`eq-level-set-transport` does not keep this property, so we restore it after each time step (see {ref}`reinitialize-the-level-set-function`).

## Notation

The mesh is made of square cells at levels between $\underline{J} = 4$ and $\overline{J} = 8$.
We write $\phi_{j, k, h}^n$ for the value of $\phi$ at time $t^n$ in the cell of level $j$ with index $k$ along $x$ and index $h$ along $y$.
On the unit square, a cell of level $j$ has side $\Delta x_j = 2^{-j}$ (`mesh.cell_length(j)`), and the finest cells have side $\Delta x_{\overline{J}} = 2^{-8}$ (`mesh.min_cell_length()`).

Values with half indices, such as $F_{j, k-1/2, h}^n$, live on the interface between two neighboring cells of the same level.
$\phi$ and $\mathbf{u}$ are stored on the leaves of the mesh (`mesh_id_t::cells`) only.
A scheme on a leaf also reads values of neighbors at the same level that are not leaves: these are ghost cells, filled by {cpp:func}`samurai::update_ghost` before each use.

For a real number $a$, we write $a^+ = \max(0, a)$ and $a^- = \min(0, a)$.

## The test case

The test case is the vortex of Bell, Colella and Glaz, as used in section 7.3 of {ref}`Min and Gibou <ref-min-gibou>` and section 4.3 of {ref}`Bellotti and Theillard <ref-bellotti>`.
The domain is $\Omega = [0, 1]^2$ and the velocity field is the steady vortex

$$
\mathbf{u}(x, y) = \left( -\sin^2(\pi x) \sin(2\pi y), \; \sin^2(\pi y) \sin(2\pi x) \right).
$$

The initial contour is the circle of radius $3/20$ centered at $(1/2, 3/4)$, given by its signed distance

$$
\phi(0, x, y) = \sqrt{(x - 1/2)^2 + (y - 3/4)^2} - 3/20.
$$

The two papers reverse the velocity at mid-time, so that the exact final contour is the initial circle.
The demo keeps the same velocity until the final time, so only the area inside the contour, which a divergence-free flow preserves, can be checked against an exact value (see {ref}`level-set-results`).

`init_level_set` sets this value at the center of each leaf and gives $\phi$ a homogeneous Neumann boundary condition:

```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
:language: c++
:start-at: template <class Field>
:end-before: template <class Mesh>
```

`init_velocity` builds $\mathbf{u}$ as a vector field with two components.
It fills the leaves and the ghost cells (`mesh_id_t::cells_and_ghosts`) and also uses a homogeneous Neumann boundary condition:

```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
:language: c++
:start-at: template <class Mesh>
:end-before: template <class Field, class Tag>
:lines: 1-22,31-
```

The velocity is computed once, on the initial mesh.
When the mesh changes, $\mathbf{u}$ moves to the new cells with the same projection and prediction operators as $\phi$ (see {ref}`adapt-the-mesh`).

The mesh is an AMR mesh with levels 4 to 8.
It starts with all its cells at level 8 (`start_level(8)`), and the first adaptation coarsens it away from the contour.
The schemes below read two cells on each side, hence `max_stencil_radius(2)`:

```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
:language: c++
:start-at: const samurai::Box<double, dim> box(min_corner, max_corner);
:end-before: double dt
:dedent:
```

The demo runs with these parameters by default:

| Parameter | Value | Command-line option |
| --- | --- | --- |
| Levels $\underline{J}$ to $\overline{J}$ | 4 to 8 | `--min-level`, `--max-level` |
| Final time $T$ | 3.14 | `--Tf` |
| CFL number | $5/8$ | `--cfl` |
| Time step $\Delta t$ | $\text{CFL} \times \Delta x_{\overline{J}} = 5/2048$ | |
| Fictitious time step $\Delta \tau_j$ on level $j$ | $\Delta x_j / 4$ | |
| Fictitious iterations per time step | 5 | |
| Boundary conditions | homogeneous Neumann for $\phi$ and $\mathbf{u}$ | |

Since $\left| u \right| \leq 1$ and $\left| v \right| \leq 1$, the time step keeps $\left| u \right| \Delta t / \Delta x_j \leq 5/8$ at every level.
All levels advance with the same time step $\Delta t$.

## Transport the level-set function

The papers cited on this page transport the level-set function with a semi-Lagrangian method.
The demo discretizes {eq}`eq-level-set-transport` with an explicit finite volume scheme instead.
Since $\mathbf{u}$ is divergence-free, {eq}`eq-level-set-transport` is also the conservation law $\partial_t \phi + \nabla \cdot (\phi \mathbf{u}) = 0$.
On each leaf, with $F$ the numerical flux along $x$ and $G$ the numerical flux along $y$:

$$
\phi_{j, k, h}^{n+1} = \phi_{j, k, h}^{n} - \frac{\Delta t}{\Delta x_j} \left( F_{j, k+1/2, h}^n - F_{j, k-1/2, h}^n + G_{j, k, h+1/2}^n - G_{j, k, h-1/2}^n \right).
$$ (eq-level-set-fv)

The scheme uses the finite volume framework of {{ project }} (see the {doc}`finite volume schemes reference <../reference/finite_volume_schemes>`): the demo defines the numerical flux on a face, and {{ project }} computes {eq}`eq-level-set-fv` on every leaf.
Each flux is computed once per face and added to the two leaves that share it, with opposite signs.
Where two leaves of different levels meet, the flux is computed between two cells of the finer level, one of them a ghost cell, and the coarse leaf receives the fluxes through its fine faces, so the scheme stays conservative across level jumps.

### The limited Lax-Wendroff flux

The flux is the limited Lax-Wendroff flux of {ref}`LeVeque <ref-leveque>` (equation 6.32), with a limiter $\psi : \mathbb{R} \to [0, 2]$.
On the face between the cells $(j, k, h)$ and $(j, k+1, h)$:

$$
\begin{aligned}
F_{j, k+1/2, h}^n = {} & \left( u_{j, k+1/2, h}^n \right)^+ \phi_{j, k, h}^n + \left( u_{j, k+1/2, h}^n \right)^- \phi_{j, k+1, h}^n \\
& + \frac{1}{2} \left| u_{j, k+1/2, h}^n \right| \left( 1 - \left| u_{j, k+1/2, h}^n \right| \frac{\Delta t}{\Delta x_j} \right) \psi\left( \theta_{j, k+1/2, h}^n \right) \left( \phi_{j, k+1, h}^n - \phi_{j, k, h}^n \right).
\end{aligned}
$$ (eq-level-set-flux)

The first line is the upwind flux and the second line the limited correction.
The limiter is the monotonized central (MC) limiter of {ref}`van Leer <ref-van-leer>`:

$$
\psi(\theta) = \max\left( 0, \min\left( 2\theta, \frac{1 + \theta}{2}, 2 \right) \right).
$$

Following equation 6.36 of {ref}`LeVeque <ref-leveque>`, $\theta$ is the ratio of the jump at the upwind face to the jump at the current face:

$$
\theta_{j, k+1/2, h}^n =
\begin{cases}
\dfrac{\phi_{j, k, h}^n - \phi_{j, k-1, h}^n}{\phi_{j, k+1, h}^n - \phi_{j, k, h}^n} & \text{if } u_{j, k+1/2, h}^n \geq 0, \\[2ex]
\dfrac{\phi_{j, k+2, h}^n - \phi_{j, k+1, h}^n}{\phi_{j, k+1, h}^n - \phi_{j, k, h}^n} & \text{if } u_{j, k+1/2, h}^n < 0.
\end{cases}
$$

When the denominator is smaller than $10^{-8}$ in absolute value, the code replaces it with $10^{-8}$.

The velocity on the face is the cubic interpolation of the first velocity component in the cells $k-1$ to $k+2$:

$$
u_{j, k+1/2, h}^n = \frac{1}{16} \left( -u_{j, k-1, h}^n + 9 u_{j, k, h}^n + 9 u_{j, k+1, h}^n - u_{j, k+2, h}^n \right).
$$

The flux on a face therefore reads the four cells $k-1$ to $k+2$.
$G$ is built the same way along $y$, with the second velocity component $v$ and the indices $h - 1$ to $h + 2$.
As {ref}`LeVeque <ref-leveque>` points out (p. 163), we cannot expect this finite volume scheme to be formally second-order accurate when the velocity field $\mathbf{u}$ is not uniform.

### The flux in the code

`make_level_set_convection` builds the scheme:

```{literalinclude} ../../../demos/FiniteVolume/level_set_schemes.hpp
:language: c++
:start-at: template <class Field, class VelocityField>
:end-before: /**
```

- `FluxConfig` declares a non-linear flux (`SchemeType::NonLinear`) on a stencil of 4 cells. It reads the scalar field $\phi$, returns a scalar field, and takes the velocity field as a parameter field.
- With 4 cells, the stencil in the direction `d` is the default one, the offsets $-1$ to $2$ along `d`: `phi[0]` to `phi[3]` are $\phi_{j, k-1, h}$ to $\phi_{j, k+2, h}$, and the face lies between `phi[1]` and `phi[2]` (see the diagram below). `data.cells` holds the same cells, in which the lambda reads the velocity.
- `data.cell_length` is $\Delta x_j$, the length of the cells the flux is computed with, which is the finer level at a level jump.
- `cons_flux_function` defines a conservative flux: the value that leaves one cell enters the other.
- `dt` is captured by reference, because the demo shortens the last time step to stop at $T$.
- `set_parameter_field(u)` makes the scheme fill the ghost cells of $\mathbf{u}$ before it is applied, when they are not up to date.

Each of the four cells has three names: its indices in the math, the value the lambda receives and its offset in the stencil.
The cell at offset `{0,0}`, on the left of the face, is the origin of the stencil:

```{diagram}
:figure: flux_stencil

The four cells of the flux $F_{j, k+1/2, h}$ along $x$, with the three indices of each.
The origin of the stencil is shaded and hatched in [red]{.sm-red} along its sides.
Besides the velocity, the flux reads the offsets $[-1, 2)$, `phi[0]` to `phi[2]`, when $u_{j, k+1/2, h} \geq 0$, and the offsets $[0, 3)$, `phi[1]` to `phi[3]`, when $u_{j, k+1/2, h} < 0$.
```

Along $y$, the offsets are `{0,-1}` to `{0,2}` and `phi[0]` to `phi[3]` are $\phi_{j, k, h-1}$ to $\phi_{j, k, h+2}$.

The demo builds the scheme once, before the time loop:

```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
:language: c++
:start-at: auto phi_0 = samurai::make_scalar_field
:end-at: make_level_set_reinitialization(phi_0);
:dedent:
```

At each time step, `convection(phi)` returns a field that holds, on each leaf, the term in parentheses of {eq}`eq-level-set-fv` divided by $\Delta x_j$:

```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
:language: c++
:start-at: // Numerical scheme
:end-at: std::swap(phi.array(), phinp1.array());
:dedent:
```

(reinitialize-the-level-set-function)=

## Reinitialize the level-set function

After each transport step, we bring $\phi$ back to a signed distance function with the reinitialization equation of Sussman, Smereka and Osher, discretized as in section 6 of {ref}`Min and Gibou <ref-min-gibou>`.
Let $\phi^0$ be the level-set function right after the transport step.
We solve the eikonal equation in a fictitious time $\tau$:

$$
\begin{cases}
\partial_{\tau} \phi + S(\phi^0) \left( \left| \nabla \phi \right| - 1 \right) = 0, \\
\phi(\tau = 0) = \phi^0,
\end{cases}
$$

where $S(\phi^0) = 1$ if $\phi^0 \geq 0$ and $S(\phi^0) = -1$ otherwise.
At steady state, $\left| \nabla \phi \right| = 1$.
The equation does not change the sign of $\phi$, hence the contour, but its discretization moves the contour slightly at each iteration.
Min and Gibou prevent this with the correction of Russo and Smereka, which uses the position of the contour in the one-sided differences of the cells it crosses; the demo does not implement it.

Only the steady state matters, so the demo does not iterate to it: it makes 5 fictitious iterations per time step, the number Min and Gibou find enough in practice (section 6.1).

### The Godunov Hamiltonian

The semi-discretization uses the Godunov Hamiltonian.
Its arguments $a$ and $b$ are the backward and forward differences along $x$, and $c$ and $d$ the backward and forward differences along $y$:

$$
H(a, b, c, d) =
\begin{cases}
\sqrt{\max\left( (a^+)^2, (b^-)^2 \right) + \max\left( (c^+)^2, (d^-)^2 \right)} - 1 & \text{if } \phi^0 \geq 0, \\[1ex]
-\left( \sqrt{\max\left( (a^-)^2, (b^+)^2 \right) + \max\left( (c^-)^2, (d^+)^2 \right)} - 1 \right) & \text{if } \phi^0 < 0.
\end{cases}
$$

The demo uses the second-order one-sided differences of {ref}`Min and Gibou <ref-min-gibou>` (equation 10):

$$
D_x^- \phi_{j, k, h} = \frac{\phi_{j, k, h} - \phi_{j, k-1, h}}{\Delta x_j} + \frac{\Delta x_j}{2} \operatorname{minmod}\left( D_{xx} \phi_{j, k, h}, D_{xx} \phi_{j, k-1, h} \right),
$$

$$
D_x^+ \phi_{j, k, h} = \frac{\phi_{j, k+1, h} - \phi_{j, k, h}}{\Delta x_j} - \frac{\Delta x_j}{2} \operatorname{minmod}\left( D_{xx} \phi_{j, k, h}, D_{xx} \phi_{j, k+1, h} \right),
$$

where $D_{xx} \phi_{j, k, h} = \left( \phi_{j, k+1, h} - 2 \phi_{j, k, h} + \phi_{j, k-1, h} \right) / \Delta x_j^2$ and $\operatorname{minmod}(a, b)$ is the argument of smaller absolute value if $a$ and $b$ have the same sign, and $0$ otherwise.
The minmod limiter keeps the differences stable where $\phi$ has kinks.
$D_y^-$ and $D_y^+$ are built the same way along $y$.
We write $H(\phi) = H\left( D_x^- \phi, D_x^+ \phi, D_y^- \phi, D_y^+ \phi \right)$.

### The TVD-RK2 time stepping

The fictitious time stepping is the second-order total variation diminishing Runge-Kutta scheme (TVD-RK2).
From $\phi^{\eta}$ to $\phi^{\eta + 1}$:

$$
\begin{aligned}
\overline{\phi}_{j, k, h} &= \phi_{j, k, h}^{\eta} - \Delta \tau \, H\left( \phi^{\eta} \right)_{j, k, h}, \\
\overline{\overline{\phi}}_{j, k, h} &= \overline{\phi}_{j, k, h} - \Delta \tau \, H\left( \overline{\phi} \right)_{j, k, h}, \\
\phi_{j, k, h}^{\eta + 1} &= \frac{1}{2} \phi_{j, k, h}^{\eta} + \frac{1}{2} \overline{\overline{\phi}}_{j, k, h}.
\end{aligned}
$$

The reinitialization runs on every leaf, at every level.
Since only the steady state matters, the levels do not need to share a fictitious time step: a cell of level $j$ advances with $\Delta \tau_j = \Delta x_j / 4$, in the same way as Min and Gibou adapt the time step to each cell (section 6.1).
On the coarse cells, the reinitialization keeps $\phi$ close to a signed distance, which the refinement criterion relies on (see {ref}`adapt-the-mesh`).

In the demo, `phi_0` is $\phi^0$, `phihat` is $\overline{\phi}$, and the ghost cells of each stage are updated before the scheme reads them:

```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
:language: c++
:start-at: // Reinitialization of the level set
:end-before: if (t >= static_cast<double>(nsave) * dt_save
:dedent:
```

`dt_fict` is $\Delta \tau_{\overline{J}} = \Delta x_{\overline{J}} / 4$.

### The Hamiltonian in the code

$H(\phi)$ is not the divergence of a flux, so the demo writes it as a cell-based scheme: a function that computes the value on a cell from the values on a stencil around it.
It is built like the {doc}`local schemes <../reference/local_schemes>`, with a star stencil instead of a single cell:

```{literalinclude} ../../../demos/FiniteVolume/level_set_schemes.hpp
:language: c++
:start-at: template <class Field>
```

- `StarStencilSchemeConfig` declares a non-linear scheme on the star stencil of radius 2: the cell and its two neighbors on each side in each direction, in the order of the comment (see the diagram below).
- The scheme function receives the cells of the stencil and $\phi$, and sets `value` on the center cell. `d_minus` and `d_plus` are $D^- \phi$ and $D^+ \phi$ in the direction `d`, and the loop over the directions works in any dimension.
- `phi_0` is captured by reference. The demo declares it once, next to the scheme, and copies $\phi$ into it after each transport step (`phi_0 = phi`).
- The function returns $\left( \Delta x_j / \Delta x_{\overline{J}} \right) H(\phi)$, so that `dt_fict * reinitialization(phi)` is $\Delta \tau_j \, H(\phi)$ on each level.

`cells[i]` is the cell of index `i` in the stencil.
The loop over the directions finds the two cells on each side of the center along `d` from these indices:

```{diagram}
:figure: star_stencil

The star stencil of radius 2 in 2D.
Each cell holds its index in `cells`, its name in the comment and its vector; the center, index 2 (`cfg::center_index`), is shaded and hatched in [red]{.sm-red} along its sides.
The table gives the indices `m2`, `m1`, `p1` and `p2` of the cells at $-2$, $-1$, $+1$ and $+2$ along `d`.
```

(adapt-the-mesh)=

## Adapt the mesh

Before each time step, we adapt the mesh to the contour.
{ref}`Min and Gibou <ref-min-gibou>` (section 3) split a cell $C$ when $\min_{v} \left| \phi(v) \right| \leq \text{Lip}(\phi) \operatorname{diag}(C)$, where $v$ runs over the vertices of $C$, $\operatorname{diag}(C)$ is the length of its diagonal and $\text{Lip}(\phi)$ is the Lipschitz constant of $\phi$: the size of a cell follows its distance to the contour.
{ref}`Theillard et al. <ref-theillard>` (equation 32) and {ref}`Bellotti and Theillard <ref-bellotti>` (equation 13) use the same criterion and note that it can be changed to keep a band of uniform width of finest cells around the contour.

The demo uses such a band: a cell must be at the finest level when the value of $\phi$ at its center satisfies

$$
\left| \phi_{j, k, h} \right| < \text{Lip}(\phi) \, M \sqrt{2} \, \Delta x_{\overline{J}},
$$ (eq-level-set-criterion)

with $\text{Lip}(\phi) = 1.2$ and $M = 5$, two choices of the demo.
The threshold uses the diagonal $\sqrt{2} \, \Delta x_{\overline{J}}$ of the finest cells, whatever the level of the cell.
Since the reinitialization keeps $\phi$ close to a signed distance, the band extends about $8.5$ finest cells on each side of the contour.

`AMR_criteria` tags each leaf with a {cpp:enum}`samurai::CellFlag`:

- a cell that meets {eq}`eq-level-set-criterion` is tagged `refine`, or `keep` if it is already at the finest level;
- any other cell is tagged `coarsen`, or `keep` if it is already at the coarsest level.

```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
:language: c++
:start-at: template <class Field, class Tag>
:end-before: template <class Field, class Phi>
```

The adaptation loop then applies the tags until the mesh no longer changes:

```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
:language: c++
:start-at: while (true)
:end-before: t += dt;
:lines: 1-2,4-
:dedent:
```

In each pass:

1. `tag.resize()` resizes the tag field to the current mesh.
2. `AMR_criteria` sets the tags.
3. {cpp:func}`samurai::graduation` changes the tags so that the new mesh stays graded: two neighboring cells, in the directions of `stencil_grad`, differ by one level at most. This is why bands of levels 5 to 7 surround the band of level 8 in the figure below. `stencil_grad` holds the four directions $\pm x$ and $\pm y$:

   ```{literalinclude} ../../../demos/FiniteVolume/level_set_AMR.cpp
   :language: c++
   :start-at: const xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil_grad{
   :end-at: };
   :dedent:
   ```

4. {cpp:func}`samurai::update_ghost` fills the ghost cells of $\phi$ and $\mathbf{u}$, which the transfer to the new mesh reads.
5. {cpp:func}`samurai::update_field` builds the new mesh from the tags and moves $\phi$ and $\mathbf{u}$ to it. A cell tagged `refine` is split into four cells, whose values come from the prediction operator; four cells tagged `coarsen` are merged into their parent cell, whose value is the average of theirs (projection). It returns `true` when the new mesh equals the current one, which ends the loop.

A cell changes by one level at most per pass, so the loop needs several passes on the first time step to coarsen the initial mesh from level 8 to level 4.

`update_field` only moves the fields it receives.
The demo resizes the other fields on the mesh (`tag`, `phinp1`, `phihat`) with `resize()` before it uses them.

(level-set-results)=

## Results

To save the solution at $t = 1.57$ and $t = 3.14$, run the demo with two output files:

```bash
./build/demos/FiniteVolume/finite-volume-level-set-amr --nfiles 2
```

The figure shows the level of each cell (color) and the contour $\phi = 0$ (white line) in `FV_level_set_2d_AMR_ite_1.h5` ($t = 1.57$, left) and `FV_level_set_2d_AMR_ite_2.h5` ($t = 3.14$, right).

```{image} ./figures/level_set.png
:alt: Two square plots of the AMR mesh colored by level, from 4 (gray) to 8 (dark purple). On the left, at t = 1.57, the white contour is a thin curved strip that winds once around the center of the domain. On the right, at t = 3.14, the strip has wound into a longer spiral. In both plots, a band of level 8 cells follows the contour, surrounded by bands of levels 7, 6 and 5, and the rest of the domain is at level 4.
:width: 100%
:align: center
```

The flow stretches the circle into a thin spiral.
The finest cells stay in a band around the contour, and the mesh stays coarse (level 4) in the rest of the domain.

The velocity field is divergence-free, so the area inside the contour stays equal to $\pi (3/20)^2 \approx 0.0707$.
At $t = 3.14$, the cells where $\phi < 0$ cover an area of $0.0699$, $1.1\,\%$ less.

## What we built

We transported a level-set function on an AMR mesh with a limited Lax-Wendroff flux written in the finite volume framework of {{ project }}, reinitialized it on every level with a cell-based scheme for the Godunov Hamiltonian and a TVD-RK2 time stepping, and adapted the mesh to the contour with a user-defined criterion, {cpp:func}`samurai::graduation` and {cpp:func}`samurai::update_field`.

## Next steps

- [`demos/FiniteVolume/level_set_MRA.cpp`](https://github.com/hpc-maths/samurai/blob/main/demos/FiniteVolume/level_set_MRA.cpp) solves the same problem on a multiresolution mesh.
- The {doc}`graduation tutorial <graduation>` explains what graduation does to a mesh.
- The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` describes the other flux types and the ready-made operators of {{ project }}, such as `make_convection_upwind` and `make_convection_weno5`.

## References

(ref-osher-fedkiw)=

S. Osher and R. Fedkiw.
*Level Set Methods and Dynamic Implicit Surfaces*.
Applied Mathematical Sciences 153, Springer, 2003.
[doi:10.1007/b98879](https://doi.org/10.1007/b98879)

(ref-leveque)=

R. J. LeVeque.
*Finite Volume Methods for Hyperbolic Problems*.
Cambridge University Press, 2002.
[doi:10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)

(ref-van-leer)=

B. van Leer.
Towards the ultimate conservative difference scheme. IV. A new approach to numerical convection.
*Journal of Computational Physics*, 23(3):276-299, 1977.
[doi:10.1016/0021-9991(77)90095-X](https://doi.org/10.1016/0021-9991(77)90095-X)

(ref-gibou-2018)=

F. Gibou, R. Fedkiw and S. Osher.
A review of level-set methods and some recent applications.
*Journal of Computational Physics*, 353:82-109, 2018.
[doi:10.1016/j.jcp.2017.10.006](https://doi.org/10.1016/j.jcp.2017.10.006)

(ref-min-gibou)=

C. Min and F. Gibou.
A second order accurate level set method on non-graded adaptive Cartesian grids.
*Journal of Computational Physics*, 225(1):300-321, 2007.
[doi:10.1016/j.jcp.2006.11.034](https://doi.org/10.1016/j.jcp.2006.11.034)

(ref-theillard)=

M. Theillard, F. Gibou and D. Saintillan.
Sharp numerical simulation of incompressible two-phase flows.
*Journal of Computational Physics*, 391:91-118, 2019.
[doi:10.1016/j.jcp.2019.04.024](https://doi.org/10.1016/j.jcp.2019.04.024)

(ref-bellotti)=

T. Bellotti and M. Theillard.
A coupled level-set and reference map method for interface representation with applications to two-phase flows simulation.
*Journal of Computational Physics*, 392:266-290, 2019.
[doi:10.1016/j.jcp.2019.05.003](https://doi.org/10.1016/j.jcp.2019.05.003)
