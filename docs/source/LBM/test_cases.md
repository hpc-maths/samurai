# Lattice Boltzmann test cases with multiresolution

This page lists the test cases that validate the lattice Boltzmann method coupled with multiresolution (MR-LBM) in {{ project }}.
For each case it gives the target equations, the initial and boundary conditions, the parameters, and the programs of `demos/LBM` that implement it.

The one-dimensional cases come from {ref}`Bellotti et al. (2022a) <ref-lbm-mr-1d>`.
The Lax-Liu problem and the von Kármán street come from {ref}`Bellotti et al. (2022b) <ref-lbm-mr-2d>`.
The figures are reproduced from these papers.

Two families of programs live in `demos/LBM`:

- the `new_*.cpp` programs use the `schemes/lbm` formalism (`#include <samurai/schemes/lbm.hpp>`).
  They are built with the demos, as targets named `lbm-new-*`.
  Their default parameters do not always match the paper: each section lists the differences.
- the other programs (`test_*.cpp` and the `D*Q*_*.cpp` files) are the programs that produced the figures of the papers.
  They use an older API, and `demos/LBM/CMakeLists.txt` does not build them.

To build and run one of the `new_*.cpp` programs, configure {{ project }} with the demos enabled, build the target, and run the executable from the build tree:

```bash
cmake -S . -B build -DBUILD_DEMOS=ON
cmake --build build --target lbm-new-D1Q222-euler-sod
./build/demos/LBM/lbm-new-D1Q222-euler-sod
```

Each program prints its options with `--help` and writes its solution with `samurai::save` in the directory given by `--path`.

The test cases use six velocity sets.
The diagram below draws each set with the indices of its velocities $c_\alpha$ in the `new_*.cpp` programs, and the lattice velocity $\lambda$ of each case that uses it.
The physical velocity attached to $c_\alpha$ is $\lambda c_\alpha$.

```{diagram}
:figure: lbm_velocity_sets

The six velocity sets on their lattice nodes. Each velocity $c_\alpha$ is an arrow from the center node to the node it reaches in one time step, and the rest velocity $c_0$ is the dot named in the key under its set. The vectorial schemes juxtapose these sets: D1Q222 is three D1Q2, D2Q4444 four D2Q4, and D2Q5444 one D2Q5 and three D2Q4.
```

## D1Q2 scheme for the advection and Burgers equations

The target problem is the scalar conservation law

$$
\partial_t u + \partial_x \varphi(u) = 0,
\qquad
\varphi(u) =
\begin{cases}
3u/4 & \text{(advection)}, \\
u^2/2 & \text{(Burgers)}.
\end{cases}
$$

The D1Q2 scheme has the velocities $\pm\lambda$, with $\lambda = 1$.
The domain is $\Omega = [-3, 3]$.
The paper runs five tests:

| Test | Equation | Initial datum $u(t=0, x)$ | Final time $T$ |
| --- | --- | --- | --- |
| I | advection | $e^{-20x^2}$ | 0.4 |
| II | advection | $\chi_{\lvert x \rvert \leq 1/2}(x)$ | 0.4 |
| III | Burgers | $(1 + \tanh(100x))/2$ | 0.4 |
| IV | Burgers | $\chi_{\lvert x \rvert \leq 1/2}(x)$ | 0.7 |
| V | Burgers | $(1 + x)\,\chi_{[-1, 0)}(x) + (1 - x)\,\chi_{[0, 1)}(x)$ | 1.3 |

The following programs implement this case:

- `test_D1Q2.cpp` runs the five tests (option `--test`, from 1 to 5) and computes the errors of the paper.
- `D1Q2_Advection_and_Burgers.cpp` runs test V and produced the figure below.
- `new_D1Q2_advection.cpp` (target `lbm-new-D1Q2-advection`) runs the same scheme with the advection velocity $3/4$, or the Burgers flux with `--burgers`, which raises $\lambda$ to at least 2.
  It does not implement the five initial data above: its domain is $[-1, 1]$ with periodic boundaries, and its initial datum is $\sin(\pi x)$.

```{figure} ./figures/D1Q2.png
:width: 100%
:align: center
:alt: Two plots against x on [-3, 3]. On the left, the level of each cell: level 2 near the ends of the domain, level 4 in the middle, up to level 7 around x = -1 and up to level 9 only near x = 1.1. On the right, the solution: zero up to x = -1, a straight ramp up to about 0.95 near x = 1.1, then a jump back to zero.

Test V at $T = 1.3$: levels of the adaptive mesh (left) and solution on the cells (right).
```

## D1Q3 and D1Q5 schemes for the shallow water system

The problem is a dam break for the shallow water system:

$$
\begin{cases}
\partial_t h + \partial_x (h u) = 0, \\
\partial_t (h u) + \partial_x (h u^2 + g h^2/2) = 0, \\
h(t=0, x) = 2\,\chi_{x < 0}(x) + \chi_{x \geq 0}(x), \\
u(t=0, x) = 0,
\end{cases}
$$

with the gravity constant $g = 1$.
This initial datum develops a left-moving rarefaction fan and a right-moving shock.
The D1Q3 scheme has the velocities $\{0, \pm\lambda\}$ and the D1Q5 scheme the velocities $\{0, \pm\lambda, \pm 2\lambda\}$, with $\lambda = 2$.
The domain is $\Omega = [-1, 1]$ and the final time is $T = 0.2$.

The following programs implement this case:

- `test_D1Q3.cpp` and `test_D1Q5.cpp` compute the errors of the paper.
- `D1Q3_Shallow_Waters_Dam.cpp` and `D1Q5_Shallow_Waters_Dam.cpp` plot the solution, with $T = 0.35$ in the D1Q3 program.
- `new_D1Q3_shallow_waters_dam.cpp` (target `lbm-new-D1Q3-shallow-waters-dam`) uses the same initial datum and parameters, with $T = 1$ by default (option `--Tf`).
  Reflecting walls (bounce-back) close both ends of the domain; with `--bc antibounceback` the right end is an open reservoir at height $h = 1$.
- `new_D1Q5_shallow_waters_dam.cpp` (target `lbm-new-D1Q5-shallow-waters-dam`) uses the same initial datum and parameters, with $T = 0.35$ by default and zero-gradient boundaries.

```{figure} ./figures/D1Q5.png
:width: 100%
:align: center
:alt: Two plots against x on [-1, 1]. On the left, the level of each cell, from 3 at the ends of the domain up to 9 around the rarefaction near x = -0.2 and the shock near x = 0.4. On the right, the height h decreases from 2 to about 1.45 across the rarefaction, then drops to 1 at the shock; the discharge hu rises from 0 to about 0.6, then drops to 0 at the shock.

D1Q5 scheme (figure 6 of the paper): levels of the adaptive mesh (left) and conserved moments $h$ and $hu$ (right).
```

## Vectorial D1Q222 scheme for the Euler system

The target system is the Sod shock tube for the Euler equations:

$$
\begin{cases}
\partial_t \rho + \partial_x (\rho u) = 0, \\
\partial_t (\rho u) + \partial_x (\rho u^2 + p) = 0, \\
\partial_t E + \partial_x (E u + p u) = 0, \\
\rho(t=0, x) = \chi_{x < 0}(x) + \tfrac{1}{8}\,\chi_{x \geq 0}(x), \\
u(t=0, x) = 0, \\
p(t=0, x) = \chi_{x < 0}(x) + \tfrac{1}{10}\,\chi_{x \geq 0}(x),
\end{cases}
$$

with the pressure law $E = \rho u^2/2 + p/(\gamma - 1)$, where $\gamma = 1.4$ is the ratio of specific heats.
The initial total energy is therefore $E = 5/2$ for $x < 0$ and $E = 1/4$ for $x \geq 0$.

The scheme juxtaposes three D1Q2 schemes, one for each of $\rho$, $\rho u$ and $E$, coupled through their equilibria.
The lattice velocity is $\lambda = 3$, the domain is $\Omega = [-1, 1]$ and the final time is $T = 0.4$.

The following programs implement this case:

- `test_D1Q222.cpp` computes the errors of the paper.
- `D1Q222_Euler_Sod.cpp` plots the solution.
- `new_D1Q222_euler_sod.cpp` (target `lbm-new-D1Q222-euler-sod`) uses the same initial datum and parameters, with zero-gradient boundaries.

```{figure} ./figures/D1Q222.png
:width: 100%
:align: center
:alt: Two plots against x on [-1, 1]. On the left, the level of each cell, from 3 on the left end up to 9 across the rarefaction and at the shock near x = 0.7. On the right, the density, momentum and energy profiles of the Sod problem, with a rarefaction between x = -0.5 and 0, a contact discontinuity near x = 0.35 and a shock near x = 0.7.

D1Q222 scheme for the Sod problem (figure 9 of the paper): levels of the adaptive mesh (left) and conserved moments $\rho$, $\rho u$ and $E$ (right).
```

## Vectorial D2Q4444 scheme for the Lax-Liu Riemann problems

The two-dimensional Euler system is

```{math}
:label: lbm-euler-2d

\begin{cases}
\partial_t \rho + \partial_x (\rho u) + \partial_y (\rho v) = 0, \\
\partial_t (\rho u) + \partial_x (\rho u^2 + p) + \partial_y (\rho u v) = 0, \\
\partial_t (\rho v) + \partial_x (\rho u v) + \partial_y (\rho v^2 + p) = 0, \\
\partial_t E + \partial_x (E u + p u) + \partial_y (E v + p v) = 0,
\end{cases}
```

with the pressure law $E = \rho (u^2 + v^2)/2 + p/(\gamma - 1)$ and $\gamma = 1.4$.
The domain is $\Omega = [0, 1]^2$, with copy (zero-gradient) boundary conditions for every variable.
The initial datum is constant in each quadrant:

$$
(\rho, u, v, p)(t=0, x, y) =
\begin{cases}
(\rho_{\text{UR}}, u_{\text{UR}}, v_{\text{UR}}, p_{\text{UR}}), & x > 1/2, \quad y > 1/2, \\
(\rho_{\text{UL}}, u_{\text{UL}}, v_{\text{UL}}, p_{\text{UL}}), & x < 1/2, \quad y > 1/2, \\
(\rho_{\text{LL}}, u_{\text{LL}}, v_{\text{LL}}, p_{\text{LL}}), & x < 1/2, \quad y < 1/2, \\
(\rho_{\text{LR}}, u_{\text{LR}}, v_{\text{LR}}, p_{\text{LR}}), & x > 1/2, \quad y < 1/2.
\end{cases}
$$

The paper runs configurations 3 and 12 of {ref}`Lax and Liu <ref-lax-liu>`:

| Configuration | Quadrant | $\rho$ | $u$ | $v$ | $p$ | Final time $T$ |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | UR | 1.5 | 0 | 0 | 1.5 | 0.3 |
| 3 | UL | 0.5323 | 1.206 | 0 | 0.3 | 0.3 |
| 3 | LL | 0.138 | 1.206 | 1.206 | 0.029 | 0.3 |
| 3 | LR | 0.5323 | 0 | 1.206 | 0.3 | 0.3 |
| 12 | UR | 0.5313 | 0 | 0 | 0.4 | 0.25 |
| 12 | UL | 1 | 0.7276 | 0 | 1 | 0.25 |
| 12 | LL | 0.8 | 0 | 0 | 1 | 0.25 |
| 12 | LR | 1 | 0 | 0.7276 | 1 | 0.25 |

```{diagram}
:figure: lax_liu_configurations

The initial states of configurations 3 and 12 in $\Omega = [0, 1]^2$. Each quadrant is named as in the table, and the red arrow in a quadrant is its velocity $(u, v)$.
```

The scheme juxtaposes four D2Q4 schemes, one for each of $\rho$, $\rho u$, $\rho v$ and $E$, coupled through their equilibria, with $\lambda = 5$.

The following programs implement this case:

- `D2Q4444_Euler_Lax_Liu.cpp` (option `--config`) runs the adaptive scheme and produced the figure below.
  `D2Q4444_Euler_Lax_Liu_uniform.cpp` runs the same scheme on a uniform mesh.
- `new_D2Q4444_euler_lax_liu.cpp` (target `lbm-new-D2Q4444-euler-lax-liu`) runs configurations 3, 11, 12 and 17 (option `--riemann`, 12 by default), with the final times above and zero-gradient boundaries.

```{figure} ./figures/D2Q4444.png
:width: 100%
:align: center
:alt: Four square panels in two rows. In each row, the left panel shows the mesh levels in color with white density contours and black velocity arrows, the finest cells following the shocks and the vortex near the center of the domain; the right panel shows the relative error of the adaptive scheme, largest near the center.

Configuration 3 at $T = 0.3$ (figure 7 of the paper), with the threshold $\epsilon = 0.005$ (top) and $\epsilon = 0.001$ (bottom). Left: mesh levels, density contours (white) and velocity (black). Right: local relative error of the adaptive scheme with respect to the scheme on the uniform finest mesh.
```

## D2Q9 scheme for the incompressible Navier-Stokes equations: von Kármán street

Let $\Theta \subset \Omega$ be an obstacle, where $\Omega$ is the computational domain.
The target system is

$$
\begin{cases}
\nabla \cdot \mathbf{u} = 0, & t \geq 0, \quad \mathbf{x} \in \Omega \setminus \Theta, \\
\rho_0 \left( \partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \nabla \cdot \left( 2 \mu \dfrac{\nabla \mathbf{u} + \nabla \mathbf{u}^T}{2} \right), & t \geq 0, \quad \mathbf{x} \in \Omega \setminus \Theta, \\
\mathbf{u} = 0, & t \geq 0, \quad \mathbf{x} \in \partial \Theta, \\
\mathbf{u}(t=0, \mathbf{x}) = (u_0, 0), & \mathbf{x} \in \Omega \setminus \Theta.
\end{cases}
$$

The D2Q9 scheme of Lallemand and Luo is consistent with this system for small velocities, in a quasi-incompressible regime.
It conserves $\rho$, which stays close to $\rho_0$, and $\rho \mathbf{u}$.
The domain is $\Omega = [0, 2] \times [0, 1]$ and the obstacle is a disk.
The mesh ignores the obstacle: at the end of each time step, the distributions of a cell are blended with the equilibrium at rest, in proportion to the fraction of the cell inside the disk.
A uniform stream of velocity $(u_0, 0)$ enters through the left, top and bottom boundaries, and the right boundary is an outflow.
Both programs below add a transverse velocity $0.01\,u_0$ to the initial datum to start the vortex shedding earlier.

The following programs implement this case:

- `D2Q9_Navier_Stokes_von_Karman_street.cpp` produced the figure below: disk of radius $1/32$ centered at $(5/16, 1/2)$, Reynolds number 1200 based on the diameter, $\lambda = 1$.
- `new_D2Q9_von_karman.cpp` (target `lbm-new-D2Q9-von-karman`) uses a disk of radius $1/16$ centered at $(1/2, 1/2)$, $u_0 = 0.1$ and a Reynolds number of 100 based on the diameter (options `--radius`, `--u0` and `--Re`).
  It also prints the drag and lift coefficients.

```{figure} ./figures/D2Q9.png
:width: 100%
:align: center
:alt: Three snapshots of the channel, from top to bottom at increasing times. The colors show the mesh levels and the white lines the contours of the velocity modulus. The finest cells surround the disk near the left end and follow the vortices shed behind it, which form a street along the whole channel in the last snapshot.

Snapshots of the adaptive solution at Reynolds number 1200, at $t = 7.19$, $13.48$ and $78.39$: mesh levels (colors) and contours of the velocity modulus (white).
```

## Vectorial D2Q4444 scheme for the implosion problem

```{admonition} TODO
:class: warning

TODO: this case does not appear in the two papers cited on this page. Maintainers: which paper (with its DOI) or report presents it?
```

The system is the Euler system {eq}`lbm-euler-2d` on $\Omega = [0, 1]^2$, with $\gamma = 1.4$.
The initial datum is a gas at rest, with a low density and a low pressure in the corner $x + y \leq 1/2$:

$$
(\rho, u, v, p)(t=0, x, y) =
\begin{cases}
(1/8, 0, 0, 7/50), & x + y \leq 1/2, \\
(1, 0, 0, 1), & x + y > 1/2.
\end{cases}
$$

All the boundaries are reflecting walls.
On each wall, the momentum normal to the wall takes an anti-bounce-back condition and the other variables a bounce-back condition:

| Variable | Left and right walls | Top and bottom walls |
| --- | --- | --- |
| $\rho$ | bounce-back | bounce-back |
| $\rho u$ | anti-bounce-back | bounce-back |
| $\rho v$ | bounce-back | anti-bounce-back |
| $E$ | bounce-back | bounce-back |

```{diagram}
:figure: implosion_box

The implosion box $\Omega = [0, 1]^2$. The gas starts at rest, with the low state in the shaded corner triangle $x + y \leq 1/2$. The text beside each hatched wall lists the variables that take the anti-bounce-back and the bounce-back conditions on that wall.
```

The following programs implement this case:

- `D2Q4444_Euler_Implosion.cpp`, with $\lambda = 10$ (8 when the finest level is 8, 9.5 when it is 9) and $T = 2.5/0.3$.
- `new_D2Q4444_euler_implosion.cpp` (target `lbm-new-D2Q4444-euler-implosion`), with $\lambda = 10$ and $T = 1.5$.
  It prints the drift of the total mass, which the closed box conserves.

## Vectorial D2Q5444 scheme for the Rayleigh-Taylor instability

```{admonition} TODO
:class: warning

TODO: this case does not appear in the two papers cited on this page. Maintainers: which paper (with its DOI) or report presents it and its figure?
```

The system is the Euler system with a downward gravity of intensity $g > 0$:

$$
\begin{cases}
\partial_t \rho + \partial_x (\rho u) + \partial_y (\rho v) = 0, \\
\partial_t (\rho u) + \partial_x (\rho u^2 + p) + \partial_y (\rho u v) = 0, \\
\partial_t (\rho v) + \partial_x (\rho u v) + \partial_y (\rho v^2 + p) = -\rho g, \\
\partial_t E + \partial_x (E u + p u) + \partial_y (E v + p v) = -\rho v g,
\end{cases}
$$

with the pressure law $E = \rho (u^2 + v^2)/2 + p/(\gamma - 1)$, $\gamma = 1.4$ and $g = 2$.
The domain is $\Omega = [0, 1]^2$.
A heavy fluid of density 2 lies above a light fluid of density 1, both at rest.
The interface is $y = 1/2 + \cos(4 \pi x)/100$, and the pressure is in hydrostatic balance, with $p = 1$ on the top wall.

The scheme juxtaposes a D2Q5 scheme for $\rho$ and three D2Q4 schemes for $\rho u$, $\rho v$ and $E$, with $\lambda = 5$.
The gravity is a source term added after the collision.

All the boundaries are reflecting walls, with the conditions of the [implosion problem](#vectorial-d2q4444-scheme-for-the-implosion-problem).
The program `D2Q5444_Euler_Rayleigh_Taylor.cpp` differs for the density: it copies the outgoing distributions of the D2Q5 scheme on every wall instead of applying bounce-back.

The following programs implement this case:

- `D2Q5444_Euler_Rayleigh_Taylor.cpp`, with $T = 2.7$.
  Its energy source term is $-v g$ instead of $-\rho v g$.
- `new_D2Q5444_euler_rayleigh_taylor.cpp` (target `lbm-new-D2Q5444-euler-rayleigh-taylor`), with $T = 2$ and the source terms of the system above.

```{figure} ./figures/D2Q5444_Rayleigh_Taylor_cropped.png
:width: 100%
:align: center
:alt: Two square panels. On the left, the mesh levels from 4 to 9 in color with black velocity arrows; the finest cells follow two mushroom-shaped plumes. On the right, the density from 0.84 to 2.5: the heavy fluid on top sinks in two plumes into the light fluid, whose rising parts roll up at their tips.

Rayleigh-Taylor instability: mesh levels and velocity (left), density (right).
```

## References

(ref-lbm-mr-1d)=

T. Bellotti, L. Gouarin, B. Graille and M. Massot (2022a).
Multiresolution-based mesh adaptation and error control for lattice Boltzmann methods with applications to hyperbolic conservation laws.
*SIAM Journal on Scientific Computing*, 44(4):A2599-A2627, 2022.
[doi:10.1137/21m140256x](https://doi.org/10.1137/21m140256x)

(ref-lbm-mr-2d)=

T. Bellotti, L. Gouarin, B. Graille and M. Massot (2022b).
Multidimensional fully adaptive lattice Boltzmann methods with error control based on multiresolution analysis.
*Journal of Computational Physics*, 471:111670, 2022.
[doi:10.1016/j.jcp.2022.111670](https://doi.org/10.1016/j.jcp.2022.111670)

(ref-lax-liu)=

P. D. Lax and X.-D. Liu.
Solution of two-dimensional Riemann problems of gas dynamics by positive schemes.
*SIAM Journal on Scientific Computing*, 19(2):319-340, 1998.
[doi:10.1137/s1064827595291819](https://doi.org/10.1137/s1064827595291819)
