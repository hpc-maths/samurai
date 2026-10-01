# A finite volume solver for the 1D Burgers equation

In this tutorial, we write a {{ project }} program that solves the 1D Burgers equation with a finite volume scheme on a uniform mesh.
The initial condition is a hat function, and the simulation runs past the time at which a shock forms.
At the end, we have a program that writes the solution to an HDF5 file, and a plot to compare with the exact solution.

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
The mesh has $2^J$ cells of size

$$
\Delta x = \frac{b - a}{2^J}.
$$

The cells are

```{math}
:label: burgers-cells

C_k = [x_{k-1/2}, x_{k+1/2}], \qquad x_{k-1/2} = a + k \Delta x, \qquad k = 0, \dots, 2^J - 1,
```

with center $x_k = a + (k + 1/2) \Delta x$.
In {{ project }}, $J$ is the level of the mesh and $k$ is the index of the cell at that level.
We use $J = 8$, so $\Delta x = 6 / 256 = 0.0234375$.

### Time stepping

Time is discretized with a time step $\Delta t$, so that $t^n = n \Delta t$ for $n = 0, 1, \dots$
The unknowns are the cell averages

$$
\overline{u}_k^n \simeq \frac{1}{\Delta x} \int_{C_k} u(t^n, x) \, \text{d}x.
$$

The finite volume scheme reads

```{math}
:label: burgers-fv-scheme

\overline{u}^{n+1}_k = \overline{u}^n_k - \frac{\Delta t}{\Delta x} \left( F_{k+1/2}^n - F_{k-1/2}^n \right),
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

\overline{u}^{n+1}_k = \overline{u}^n_k - \frac{\Delta t}{2 \Delta x} \left( (\overline{u}^n_k)^2 - (\overline{u}^n_{k-1})^2 \right).
```

Another choice is the Lax-Friedrichs flux, which is more diffusive than the upwind flux:

$$
\mathcal{F}(\overline{u}_L, \overline{u}_R) = \frac{1}{2} \left( \varphi(\overline{u}_L) + \varphi(\overline{u}_R) \right) - \frac{\Delta x}{2 \Delta t} \left( \overline{u}_R - \overline{u}_L \right).
$$

### CFL condition

The time step must satisfy the CFL condition

```{math}
:label: burgers-cfl

\Delta t \leq \frac{\Delta x}{\sup_{x \in \mathbb{R}} |\varphi'(u_0(x))|}.
```

Since $\varphi'(u) = u$ and $0 \leq u \leq \sup u_0 = 1$ at all times, the condition is $\Delta t \leq \Delta x$.
We take $\Delta t = 0.99 \, \Delta x$, that is a CFL number of 0.99.
With a CFL number above 1, the scheme is unstable: oscillations appear and grow.

## Build the mesh

We start the `main` function by initializing {{ project }} and reading the command-line options:

```c++
auto& app = samurai::initialize("Burgers equation on a uniform 1D mesh", argc, argv);

double Tf  = 1.5;  // final time, after the shock forms at T* = 1
double cfl = 0.99; // CFL number: dt = cfl * dx, stable for cfl <= 1
app.add_option("--Tf", Tf, "Final time")->capture_default_str();
app.add_option("--cfl", cfl, "CFL number")->capture_default_str();
SAMURAI_PARSE(argc, argv);
```

{cpp:func}`samurai::initialize` returns the command-line application, to which we add the final time and the CFL number as options.
`SAMURAI_PARSE` reads them, together with the options of {{ project }}.

Next, we describe the mesh with {cpp:class}`samurai::mesh_config` and build it on the box $[-3, 3]$ with {cpp:func}`samurai::amr::make_mesh`:

```c++
constexpr std::size_t dim = 1;
const std::size_t level   = 8; // J: 2^8 = 256 cells

const samurai::Box<double, dim> box({-3}, {3});
auto config = samurai::mesh_config<dim>().min_level(level).max_level(level).start_level(level);
auto mesh   = samurai::amr::make_mesh(box, config);

std::cout << mesh << std::endl;
```

The mesh is an adaptive mesh refinement (AMR) mesh, but with the same minimum, maximum and start level it stays uniform: all cells are at level 8.
The level-0 cell is the whole box, so a cell at level 8 has the size $\Delta x = 6 / 2^8$.

The program prints the sub-meshes of the mesh (`cells`, `cells and ghosts`, ...).
The `cells` sub-mesh has a single interval `[0,256[` at level 8: the 256 cells $C_0, \dots, C_{255}$ of {eq}`burgers-cells`.

:::{note}
The command-line options `--min-level`, `--max-level` and `--start-level` replace the levels set in the code (see the {doc}`mesh how-to guide <../howto/mesh>`).
To keep a uniform mesh, give the same value to all three.
:::

## Set the initial condition

We create a scalar field `u` on the mesh with {cpp:func}`samurai::make_scalar_field`, and set it to the hat function {eq}`burgers-initial-condition` at the center of each cell:

```c++
auto u = samurai::make_scalar_field<double>("u", mesh);

samurai::for_each_cell(mesh,
                       [&](auto& cell)
                       {
                           const double x = cell.center(0);
                           u[cell]        = (x < -1. || x > 1.) ? 0. : 1. - std::abs(x);
                       });
```

Taking the value at the center instead of the cell average is accurate to second order in $\Delta x$ where $u_0$ is smooth.

The scheme {eq}`burgers-upwind-scheme` in the first cell $C_0$ reads $\overline{u}_{-1}$, which lies outside the domain.
{{ project }} stores such values in ghost cells, filled from the boundary conditions of the field.
We attach a homogeneous Neumann condition to `u`, which copies the value of the boundary cell into the ghost cells:

```c++
samurai::make_bc<samurai::Neumann<1>>(u, 0.);
```

The solution is zero near $x = \pm 3$ up to $T = 1.5$, so this choice does not change the result.
The {doc}`boundary conditions reference <../reference/bc>` lists the other conditions.

## Write the time loop

The scheme needs a second field for $\overline{u}^{n+1}$, and the time step from the CFL condition {eq}`burgers-cfl`:

```c++
auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

double dt      = cfl * mesh.min_cell_length(); // sup |u_0| = 1
double t       = 0.;
std::size_t nt = 0;
```

Each iteration of the loop does four things:

1. it advances the time, and shortens the last time step so that the loop stops exactly at `Tf`;
2. it fills the ghost cells of `u` with {cpp:func}`samurai::update_ghost_mr`, which also applies the boundary conditions: the values of `u` change at each time step, so the ghost cells must be updated at each time step too;
3. it applies the scheme {eq}`burgers-upwind-scheme` on each interval of cells: `u(lvl, i - 1)` is the field on the interval shifted one cell to the left, that is $\overline{u}^n_{k-1}$ for all the cells $k$ of the interval `i` at once;
4. it exchanges the values of `u` and `unp1` with {cpp:func}`samurai::swap`, so that `u` holds the new solution for the next iteration.

```c++
while (t != Tf)
{
    t += dt;
    if (t > Tf)
    {
        dt += Tf - t;
        t = Tf;
    }

    samurai::update_ghost_mr(u);

    samurai::for_each_interval(mesh,
                               [&](std::size_t lvl, const auto& i, auto)
                               {
                                   const double dx = mesh.cell_length(lvl);
                                   unp1(lvl, i)    = u(lvl, i)
                                                - 0.5 * dt / dx * (u(lvl, i) * u(lvl, i) - u(lvl, i - 1) * u(lvl, i - 1));
                               });

    samurai::swap(u, unp1);

    std::cout << "iteration " << ++nt << ": t = " << t << std::endl;
}
```

{cpp:func}`samurai::swap` exchanges the data of the two fields, and also whether their ghost cells are up to date.
The boundary condition stays attached to `u`.

After the loop, we save the mesh and the solution to `burgers_1d.h5`, in the directory the program runs from:

```c++
samurai::save("burgers_1d", mesh, u);

samurai::finalize();
return 0;
```

:::{tip}
{{ project }} also provides the upwind flux {eq}`burgers-upwind-flux` as a ready-made operator, for solutions of any sign.
`samurai::make_convection_upwind` computes the flux $u^2$, so we multiply it by $1/2$:

```c++
auto conv = 0.5 * samurai::make_convection_upwind<decltype(u)>();
unp1      = u - dt * conv(u);
```

The operator updates the ghost cells of `u` itself when they are out of date.
The demo `demos/FiniteVolume/burgers_mra.cpp` uses it on a multiresolution mesh.
The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` describes this operator and the others.
:::

## Run the program and plot the result

The complete program, with the pieces above in order:

```c++
#include <cmath>
#include <iostream>

#include <samurai/algorithm/update.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Burgers equation on a uniform 1D mesh", argc, argv);

    double Tf  = 1.5;  // final time, after the shock forms at T* = 1
    double cfl = 0.99; // CFL number: dt = cfl * dx, stable for cfl <= 1
    app.add_option("--Tf", Tf, "Final time")->capture_default_str();
    app.add_option("--cfl", cfl, "CFL number")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    // Uniform mesh with 2^8 cells on [-3, 3]
    constexpr std::size_t dim = 1;
    const std::size_t level   = 8; // J: 2^8 = 256 cells

    const samurai::Box<double, dim> box({-3}, {3});
    auto config = samurai::mesh_config<dim>().min_level(level).max_level(level).start_level(level);
    auto mesh   = samurai::amr::make_mesh(box, config);

    std::cout << mesh << std::endl;

    // Initial condition and boundary condition
    auto u = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               const double x = cell.center(0);
                               u[cell]        = (x < -1. || x > 1.) ? 0. : 1. - std::abs(x);
                           });

    samurai::make_bc<samurai::Neumann<1>>(u, 0.);

    // Time loop
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    double dt      = cfl * mesh.min_cell_length(); // sup |u_0| = 1
    double t       = 0.;
    std::size_t nt = 0;

    while (t != Tf)
    {
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        samurai::update_ghost_mr(u);

        samurai::for_each_interval(mesh,
                                   [&](std::size_t lvl, const auto& i, auto)
                                   {
                                       const double dx = mesh.cell_length(lvl);
                                       unp1(lvl, i)    = u(lvl, i)
                                                    - 0.5 * dt / dx * (u(lvl, i) * u(lvl, i) - u(lvl, i - 1) * u(lvl, i - 1));
                                   });

        samurai::swap(u, unp1);

        std::cout << "iteration " << ++nt << ": t = " << t << std::endl;
    }

    samurai::save("burgers_1d", mesh, u);

    samurai::finalize();
    return 0;
}
```

1. Build the program as described in the {doc}`CMake how-to guide <../howto/cmake>`.
2. Run it.
   After the description of the mesh, it prints one line per time step.
   With the default options, $\Delta t = 0.99 \times 6 / 256 \approx 0.0232$, and the last line is:

   ```text
   iteration 65: t = 1.5
   ```

   The program writes `burgers_1d.h5` and `burgers_1d.xdmf` in the current directory.
3. Plot the solution with the script of the {{ project }} repository, where `<samurai-dir>` is the root of your copy:

   ```bash
   python <samurai-dir>/python/read_mesh.py burgers_1d --field u
   ```

The plot shows the ramp $u = (1 + x) / 2.5$ of {eq}`burgers-solution-after-shock`, from $x = -1$ up to the shock near $x \approx 1.236$, where $u$ drops from about 0.89 to 0.
The upwind scheme spreads the shock and the corner at $x = -1$ over a few cells.
Run the program with `--max-level 10 --min-level 10 --start-level 10` to see them sharpen: $\Delta x$ is 4 times smaller, so there are 4 times more time steps.

## What we built

We solved the Burgers equation {eq}`burgers-equation` past the formation of a shock.
On the way, we built a uniform mesh from a `samurai::mesh_config`, created and initialized a scalar field, attached a boundary condition, and wrote a finite volume time loop that updates the ghost cells before each step and swaps the fields after it.

## Next steps

- The demos in [demos/tutorial/AMR_1D_Burgers](https://github.com/hpc-maths/samurai/tree/main/demos/tutorial/AMR_1D_Burgers) solve the same problem step by step and continue with adaptive mesh refinement.
- The {doc}`loop how-to guide <../howto/loop>` and the {doc}`field how-to guide <../howto/field>` cover more ways to loop over the mesh and to access fields.
- The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` describes the operators {{ project }} provides for finite volume schemes.
