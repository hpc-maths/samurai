# Reaction-diffusion equation

In this tutorial, we solve a reaction-diffusion equation with the finite volume module of {{ project }} and PETSc.
We build a diffusion operator, a non-linear reaction operator written by hand, and two implicit time schemes, then we run them on an adaptive 1D mesh.
The code comes from the demo {download}`nagumo.cpp <../../../demos/FiniteVolume/nagumo.cpp>`.

## Before you start

You need:

- PETSc and `pkg-config`, and a {{ project }} build directory configured with `-DWITH_PETSC=ON`, as described in the {doc}`installation how-to <../howto/installation>`.
  The demo is built only when PETSc is enabled.
  In your own CMake project, turn PETSc on with `SAMURAI_WITH_PETSC` (see the {doc}`CMake project how-to <../howto/cmake>`).
- To know how to create a field and set its values cell by cell, as shown in the {doc}`field tutorial <field>`.

The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` and the {doc}`local schemes reference <../reference/local_schemes>` describe the operators we use in more detail.

## The Nagumo equation

A reaction-diffusion equation has the form

$$
\partial_t u + \mathcal{D}(u) = \mathcal{R}(u),
$$ (tutorial-rd-abstract)

where $\mathcal{D}$ is a diffusion operator and $\mathcal{R}$ is a reaction operator.
We solve the Nagumo equation, whose solution is a traveling wave.
Let $\Omega \subset \mathbb{R}^d$ be the domain in dimension $d$ and $T > 0$ the final time.
We look for $u\colon [0, T] \times \Omega \to \mathbb{R}$ such that

$$
\partial_t u - D \Delta u = k u^2 (1 - u),
$$ (tutorial-rd-nagumo)

where the diffusion coefficient $D$ and the reaction coefficient $k$ are positive constants.
The solution satisfies homogeneous Neumann boundary conditions on $\partial\Omega$ and the initial condition $u = u_0$ at $t = 0$.
In the form {eq}`tutorial-rd-abstract`, the two operators are

$$
\begin{aligned}
\mathcal{D}(u) &= -D \Delta u, \\
\mathcal{R}(u) &= k u^2 (1 - u).
\end{aligned}
$$ (tutorial-rd-operators)

## Creating the mesh and the fields

We work in dimension $d = 1$ on $\Omega = [-10, 10]$, with $D = 1$ and $k = 10$:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: double left_box
  :end-at: double k = 10;
  :dedent:
```

The demo declares `dim = 1`, `Box = samurai::Box<double, dim>` and `point_t = typename Box::point_t` at the top of `main`.
We create a multiresolution mesh on the box, with cells between levels 4 and 8:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: point_t box_corner1, box_corner2;
  :end-at: samurai::mra::make_mesh(box, config);
  :dedent:
```

`disable_minimal_ghost_width()` gives the mesh one layer of ghost cells instead of the default two.
One layer is enough here: the operators of this tutorial only use the direct neighbors of a cell.

The time scheme needs the solution at two time steps, $u_n$ and $u_{n+1}$.
We store them in two scalar fields and attach the homogeneous Neumann boundary condition to both:

```c++
auto u    = samurai::make_scalar_field<double>("u", mesh);
auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

samurai::make_bc<samurai::Neumann<1>>(u);
samurai::make_bc<samurai::Neumann<1>>(unp1);
```

Without a value, {cpp:func}`samurai::make_bc` sets the boundary value to zero.

:::{remark}
The demo declares `u`, `unp1` and `rhs` with `samurai::make_vector_field<double, n_comp>` and `n_comp = 1`.
This tutorial uses scalar fields, the natural type for a scalar equation.
Of the code shown on this page, only the Jacobian function depends on this choice.
:::

## Setting the initial condition

The Nagumo equation has an exact traveling wave solution: a front that starts at $z_0 = -2$ and moves to the right at speed $c = \sqrt{k D / 2}$.
We code it as a function of $x$ and $t$:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: double z0
  :end-before: if (restart_file.empty())
  :dedent:
```

We use it at $t = 0$ as the initial condition $u_0$:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-after: // Initial solution
  :end-at: });
  :dedent:
```

Since the exact solution is known at all times, the demo also computes the $L^2$ error at each time step.

## Building the operators

### Diffusion and identity

The finite volume module provides the order 2 diffusion operator.
We create it, and the identity operator that the implicit schemes need, for the type of `u`:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: auto diff = samurai::make_diffusion_order2
  :end-at: auto id   = samurai::make_identity
  :dedent:
```

The field type is a template parameter, and the diffusion coefficient $D$ is a function argument.
In an implicit scheme, `id` produces the identity matrix.

:::{important}
`diff` is $\mathcal{D} = -D \Delta$, with the minus sign.
{cpp:func}`samurai::make_laplacian_order2` gives the Laplacian $\Delta$ without the minus sign, so this definition of `diff` is equivalent:

```c++
auto diff = -D * samurai::make_laplacian_order2<decltype(u)>();
```

:::

### Reaction

The reaction operator $\mathcal{R}$ only uses the value of `u` in the current cell: it is a local scheme, built with the {doc}`local schemes framework <../reference/local_schemes>`.
We first declare a configuration type that holds its static properties, then create the operator and give it a name:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: using cfg
  :end-at: react.set_name
  :dedent:
```

The three template parameters of `samurai::LocalCellSchemeConfig` are:

- `samurai::SchemeType::NonLinear`: the operator is non-linear;
- the type of the output field, here the type of `u`;
- the type of the input field, here the type of `u` again.

Next, we write the formula of $\mathcal{R}$ in the scheme function:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: react.set_scheme_function(
  :end-before: react.set_jacobian_function(
  :dedent:
```

The function receives:

- `value`: the result of the operator in the cell, to fill. Its type `samurai::SchemeValue<cfg>` follows the output field: a `double` for a scalar field.
- `cell`: the current cell.
- `field`: the input field. We declare it `const auto&` because {{ project }} also calls this function with a local copy of the cell value when it solves the implicit reaction cell by cell.

To use `react` in an implicit scheme, we also give its Jacobian $\mathcal{R}'(u) = k (2 u (1 - u) - u^2)$.
An explicit use of the operator does not need it.
For a scalar field, the Jacobian is a `double`:

```c++
react.set_jacobian_function(
    [&](samurai::JacobianMatrix<cfg>& jac, const auto& cell, const auto& field)
    {
        auto v = field[cell];
        jac    = k * (2 * v * (1 - v) - v * v);
    });
```

With the one-component vector field of the demo, `jac` is a $1 \times 1$ matrix instead, which the demo fills component by component.

## Choosing the time step

We discretize in time with the Euler scheme and a time step $\Delta t$.
By default, the demo uses $\Delta t = T / 100$, with $T = 1$.
When the diffusion is explicit (option `--explicit-diffusion`), the time step must satisfy a stability condition, and the demo computes it from the smallest cell length $\Delta x$ and the CFL number:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: if (dt == 0)
  :end-before: auto MRadaptation
  :dedent:
```

`Tf`, `dt`, `cfl`, `explicit_diffusion` and `explicit_reaction` are command-line options of the demo, declared at the top of `main`.
The option `--dt` sets the time step directly.

## Adapting the mesh

Multiresolution adapts the mesh to `u`: it removes the fine cells where `u` is smooth and keeps them around the front.
We create the adaptation object and adapt the mesh to the initial condition once before the time loop:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: auto MRadaptation
  :end-at: MRadaptation(mra_config);
  :dedent:
```

The threshold `epsilon` controls how many fine cells are kept: a smaller value keeps more of them.
We also need a field for the right-hand side of the linear system:

```c++
auto rhs = samurai::make_scalar_field<double>("rhs", mesh);
```

## Creating the solvers

### Implicit diffusion, explicit reaction

With implicit diffusion and explicit reaction, the Euler scheme reads

$$
u_{n+1} + \Delta t\, \mathcal{D}(u_{n+1}) = u_n + \Delta t\, \mathcal{R}(u_n).
$$ (tutorial-rd-imex)

We compute $u_{n+1}$ by solving the linear system $(\mathrm{Id} + \Delta t\, \mathcal{D})\, u_{n+1} = u_n + \Delta t\, \mathcal{R}(u_n)$, where $\mathrm{Id}$ is the identity.
We create the PETSc linear solver once, before the time loop:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: auto implicit_diffusion_solver
  :end-before: auto implicit_reaction_solver
  :dedent:
```

`id + dt * diff` is a new operator built from `id` and `diff`.
It is not a matrix: the solver assembles the PETSc matrix when it solves.
`samurai::petsc::make_solver` chooses a linear solver because the operator is linear.
The `configure` function sets a direct LU solve; it is optional.
Without it, the solver keeps the PETSc defaults described in the [PETSc KSP documentation](https://petsc.org/release/manual/ksp/).

### Implicit diffusion and reaction

With both operators implicit, the Euler scheme reads

$$
u_{n+1} + \Delta t\, \mathcal{D}(u_{n+1}) - \Delta t\, \mathcal{R}(u_{n+1}) = u_n.
$$ (tutorial-rd-implicit)

This time, $(\mathrm{Id} + \Delta t\, \mathcal{D} - \Delta t\, \mathcal{R})(u_{n+1}) = u_n$ is a non-linear system.
`react` is declared with `samurai::SchemeType::NonLinear`, so the combined operator is non-linear too, and `samurai::petsc::make_solver` returns a non-linear solver (PETSc SNES).
We configure it to use Newton's method with a line search, and an LU solve for each Newton step:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: auto full_implicit_solver
  :end-before: //----
  :dedent:
```

PETSc reads its command-line options after the `configure` function, so you can override these choices when you run the program.
For example, add `-snes_monitor` to print the residual of each Newton iteration.

## Writing the time loop

At each iteration, we advance the time and adapt the mesh to the current solution.
If the last step would go past $T$, we shorten it so that the loop ends exactly at $t = T$, and we record the change in `dt_has_changed`, a flag set to `false` before the loop:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: while (t != Tf)
  :end-at: rhs.resize();
  :dedent:
```

When the mesh changes, `unp1` and `rhs` must be resized to the new mesh.
The adaptation already updates `u`.

With implicit diffusion and explicit reaction, we compute the right-hand side, then solve the linear system {eq}`tutorial-rd-imex`:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: else if (!explicit_diffusion && explicit_reaction)
  :end-before: else if (explicit_diffusion && !explicit_reaction)
  :dedent:
```

Two calls in this branch update the solver:

- `set_scheme` gives the solver the new operator when the last time step was shortened, since the operator contains `dt`.
- `reset` makes the solver assemble a new matrix. The demo calls it at every iteration when the mesh has more than one level, since the adaptation may have changed the number of cells.

With both operators implicit, Newton's method needs an initial guess.
We start from the solution at the current time step, which is close to the solution we look for, then solve {eq}`tutorial-rd-implicit`:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-after: // Implicit diffusion and reaction
  :end-before: // u <-- unp1
  :dedent:
```

The demo also handles fully explicit time stepping and explicit diffusion with implicit reaction, in the other branches of the same `if`.
After the solve, {cpp:func}`samurai::swap` exchanges the values of `u` and `unp1` without copying them, so that `u` holds the new solution:

```{literalinclude} ../../../demos/FiniteVolume/nagumo.cpp
  :language: c++
  :start-at: // u <-- unp1
  :end-at: samurai::swap(u, unp1);
  :dedent:
```

## Running the demo

From the {{ project }} source directory, build and run the demo with its default options, which solve the fully implicit scheme {eq}`tutorial-rd-implicit`:

```bash
cmake --build build --target finite-volume-nagumo
./build/demos/FiniteVolume/finite-volume-nagumo
```

The demo prints one line per time step, with the time, the time step and the $L^2$ error against the exact solution:

```text
iteration 0: t = 0.01, dt = 0.01, L2-error: ...
iteration 1: t = 0.02, dt = 0.01, L2-error: ...
...
```

In the current directory, it writes the solution and the cell levels after each time step in `nagumo_ite_<n>.h5` and `nagumo_ite_<n>.xdmf`, where `nagumo_ite_0` holds the initial condition.
It also writes the restart files `nagumo_restart_ite_<n>.h5`.
The option `--save-final-state-only` writes only the final solution.
To plot them, follow the {doc}`plot how-to <../howto/plot>`.

{ref}`plate-nagumo-result` draws the first and the last of these files: `nagumo_ite_0` at $t = 0$ and `nagumo_ite_100` at $t = 1$, after 100 time steps.
The front moves to the right, and the finest cells move with it.

```{plate} The traveling wave and its mesh at t = 0 and t = 1
:figure: nagumo_result
:label: plate-nagumo-result

**Fig. 1.** The file `nagumo_ite_0`, at $t = 0$, after the first mesh adaptation.
Above, `u`, one [red]{.sm-red} step per cell, over the exact traveling wave of the demo.
At this scale the steps follow the wave to within a line width.
The inset at the top right enlarges 5 times in $x$ the thin rectangle drawn around the front, over the 16 cells of level 8 of $[-2.578125, -1.328125)$, with its own $u$ axis on its right: there the steps of `u` stand apart from the wave.
*a*, in the inset, the middle of the front, where the exact solution equals $1/2$: $x = z_0 = -2$.
Below, the 112 cells of the mesh, one row per level from level 4 to level 8; a dashed line runs where a level has no cell.
The 76 cells of level 8, hatched in [red]{.sm-red}, cover $[-5, 0.9375)$.
The whole front lies inside: the exact solution falls from 0.99 to 0.01 between $x \approx -4.06$ and $x \approx 0.06$.

**Fig. 2.** The file `nagumo_ite_100`, at $t = 1$.
The inset enlarges the rectangle around the front as in Fig. 1, over the cells of $[-0.3125, 0.9375)$.
*b*, the middle of the front has moved by $c = \sqrt{5} \approx 2.236$, to $x \approx 0.236$.
The mesh still has 112 cells, with the same number per level, and the cells of level 8 cover $[-2.8125, 3.125)$, around the front, which spans $x \approx -1.82$ to $x \approx 2.29$.
```

To run the scheme {eq}`tutorial-rd-imex` with explicit reaction instead, add the option:

```bash
./build/demos/FiniteVolume/finite-volume-nagumo --explicit-reaction
```

`--help` lists the other options of the demo, such as `--Tf`, `--D` and `--k`.

## What we built

We solved the Nagumo equation on an adaptive 1D mesh with two implicit Euler schemes.
On the way, we combined built-in operators with a reaction operator written as a local scheme, gave it a Jacobian, and solved the resulting linear and non-linear systems with PETSc.

## Next steps

- The {doc}`local schemes reference <../reference/local_schemes>` describes the linear local schemes and their coefficient functions.
- The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` lists the other operators, such as convection, gradient and divergence.
- The {doc}`plot how-to <../howto/plot>` shows how to plot the 1D results.
