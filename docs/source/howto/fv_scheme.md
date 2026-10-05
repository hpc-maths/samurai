# How-to: write a finite volume scheme

This guide shows you how to write your own flux-based finite volume scheme, apply it to a field and check that it computes what you expect.
You describe the flux through one face in a `samurai::FluxDefinition`, build the scheme with `samurai::make_flux_based_scheme`, and {{ project }} applies the flux to every face of the mesh, level jumps included.

## Before you start

- You know the scheme you want to implement, written as a flux through the faces of the cells.
  The {ref}`finite volume principle <fv-principle>` section of the finite volume reference explains how a differential term becomes a sum of fluxes, and which sign convention {{ project }} uses.
- You have a mesh and a field, with boundary conditions attached to the field if the mesh is not periodic in every direction. If not, see the [mesh](mesh.md), [field](field.md) and [boundary conditions](bc.md) how-to guides.
- Include `samurai/schemes/fv.hpp`.
- Check the {ref}`available implementations <fv-available-implementations>` first: diffusion, convection (upwind and WENO5), gradient, divergence and identity are already there.

The examples of this guide live in `docs/source/howto/snippet/fv_scheme/`.
Each one writes a scheme on a uniform 2D mesh of the unit square and compares it with a built-in scheme or with the exact derivative.

## Choose the scheme type

The scheme type tells {{ project }} how the flux depends on the field, and decides the signature of the flux function you write:

| The flux is | Scheme type | The flux function returns |
| --- | --- | --- |
| linear in the field, with coefficients that depend on the cell length only | `SchemeType::LinearHomogeneous` | the coefficients of the stencil values |
| linear in the field, with coefficients that vary in space | `SchemeType::LinearHeterogeneous` | the coefficients of the stencil values |
| non-linear in the field | `SchemeType::NonLinear` | the value of the flux |

A linear scheme gives its flux as coefficients rather than as a value, so that the same definition serves an explicit application and the assembly of a matrix for an implicit solve.

## Write a linear scheme

The example `fv_laplacian.cpp` writes the Laplacian $\Delta u$ as the divergence of the normal gradient, with the flux $(u_R - u_L)/h$ through each face:

```{literalinclude} snippet/fv_scheme/fv_laplacian.cpp
  :language: c++
  :start-at: // Laplacian as the divergence
  :end-before: // Largest
```

1. Declare the configuration with `samurai::FluxConfig`: the scheme type, the stencil size (the number of cells the flux reads), the type of the output field and the type of the input field.
   The examples write `using samurai::SchemeType;` at the top of the file to shorten the scheme type.
2. Create a `samurai::FluxDefinition<cfg>` and give it the flux function.
   Passing the function to the constructor sets it in every direction.
3. Build the scheme with `samurai::make_flux_based_scheme`.
   `set_name` gives it the name that appears in error messages and timers.

The flux function of a `LinearHomogeneous` scheme receives the array `c` of coefficients and the cell length `h`, and sets one coefficient per stencil cell: the flux is $\sum_i c_i u_i$.
Without a stencil of your own, the stencil of size 2 is the cell on the left of the face (index 0) and the cell on its right (index 1) in the x-direction, the bottom and top cells in the y-direction.
{{ project }} adds the flux to the cell on the left of the face and subtracts it from the cell on the right, each divided by the cell measure, so the scheme computes the cell average of $\Delta u$.

Apply the scheme like a function:

```{literalinclude} snippet/fv_scheme/fv_laplacian.cpp
  :language: c++
  :start-at: auto lap_u = my_lap(u);
  :end-at: auto lap_u = my_lap(u);
  :dedent:
```

The call updates the ghosts of `u` if they are out of date, which applies its boundary conditions, and returns a new field.

## Check the scheme

Compare your scheme with a built-in scheme that computes the same thing, and with the exact derivative of a smooth function on a sequence of meshes.
`fv_laplacian.cpp` takes $u = \sin(\pi x)\sin(\pi y)$, whose Laplacian is $-2\pi^2 u$, with a Dirichlet condition of value 0, and prints at each level the largest difference with `samurai::make_laplacian_order2`, the largest error against $-2\pi^2 u$ at the cell centers, and the order of convergence:

```{literalinclude} snippet/fv_scheme/fv_laplacian.cpp
  :language: c++
  :start-at: double diff
  :end-at: double error
  :dedent:
```

```bash
./fv_laplacian
```

```{literalinclude} snippet/fv_scheme/fv_laplacian_output.txt
  :language: text
```

The scheme matches the built-in one to the last bit, and the error is divided by 4 each time the cell length is halved: the scheme is of order 2.
A wrong sign or a wrong coefficient makes the first column nonzero.

## Map a field to a field of another size

The output and input fields can have different numbers of components.
The coefficient `c[i]` is then a matrix of size (output components) x (input components), an `xtensor` object that you fill with `fill` and `c[i](row, col)`.
When both fields are scalar fields, `c[i]` is a plain number, as in the Laplacian above.

The example `fv_gradient.cpp` computes the gradient of a scalar field, with the flux $\frac{u_L + u_R}{2}\,\mathbf{n}$ through each face:

```{literalinclude} snippet/fv_scheme/fv_gradient.cpp
  :language: c++
  :start-at: // Gradient of a scalar field
  :end-before: // Largest
```

- The output field type is a vector field with `dim` components on the same mesh: `c[i]` is a `dim` x 1 matrix.
- The flux differs from one direction to the other, so the example sets `cons_flux_function` direction by direction.
  `samurai::static_for<0, dim>` gives the direction `d` as a compile-time constant, which the flux function uses as an index.

For a divergence, the input field has `dim` components and the output field one: `c[i]` is a 1 x `dim` matrix and the coefficient goes in column `d` instead of row `d`.
Code that must work for scalar and vector fields tests `Field::is_scalar` with `if constexpr`, because a scalar `c[i]` has no `fill` and no `(row, col)` access.

The program compares the result with `samurai::make_gradient_order2`:

```{literalinclude} snippet/fv_scheme/fv_gradient_output.txt
  :language: text
```

## Make the coefficients vary in space

When the coefficients depend on the position, for example through a velocity field, use `SchemeType::LinearHeterogeneous`.
The flux function then receives the stencil data instead of the cell length, with the cells of the stencil in `data.cells` and their length in `data.cell_length`.
The example `fv_upwind.cpp` writes the upwind scheme for $\nabla \cdot (\mathbf{a} u)$ with a velocity field $\mathbf{a}$:

```{literalinclude} snippet/fv_scheme/fv_upwind.cpp
  :language: c++
  :start-at: // Upwind scheme
  :end-before: int main
```

- The fifth template argument of `FluxConfig` declares the type of a parameter field, here the velocity.
  `set_parameter_field(a)` registers the field itself: the scheme then updates the ghosts of `a` before it reads them, as it does for `u`.
  Without it, update the ghosts of `a` yourself after each change, with `samurai::update_ghost_mr(a)`.
- The flux function reads `a` in the cells of the stencil.
  Where two levels meet, these cells include ghosts, which is why the ghosts of `a` must hold values.
- The lambda captures `a` by reference: `a` must outlive the scheme.
  Capture by value any local variable of the function that builds the scheme, since the scheme outlives that function.

The program compares the result with `samurai::make_convection_upwind`, which uses the same velocity on the face:

```{literalinclude} snippet/fv_scheme/fv_upwind_output.txt
  :language: text
```

## Write a non-linear scheme

With `SchemeType::NonLinear`, the flux function computes the flux itself from the values of the field in the stencil.
The example `fv_burgers.cpp` writes $\nabla \cdot f(u)$ with the Burgers flux $f(u) = u^2/2$ in every direction and the centered flux $\frac{f(u_L) + f(u_R)}{2}$:

```{literalinclude} snippet/fv_scheme/fv_burgers.cpp
  :language: c++
  :start-at: using cfg
  :end-before: // The same flux
```

The flux function receives the flux to set, the stencil data and the stencil values `u`, where `u[i]` is the value in the `i`-th stencil cell.
For a field with several components, `u[i]` and the flux are arrays of components.

An implicit solve of a non-linear scheme runs Newton's method, which needs the derivative of the flux with respect to each stencil value: set it in `cons_jacobian_function`.
`jac[i]` is the derivative with respect to `u[i]`, a matrix of size (output components) x (input components), or a number for scalar fields.
An explicit scheme does not need it.

Set the functions in every direction and build the scheme:

```{literalinclude} snippet/fv_scheme/fv_burgers.cpp
  :language: c++
  :start-at: samurai::FluxDefinition<cfg> centered;
  :end-before: // u = sin
  :dedent:
```

```{literalinclude} snippet/fv_scheme/fv_burgers.cpp
  :language: c++
  :start-at: auto burgers =
  :end-at: burgers.set_name
  :dedent:
```

To check the Jacobian function without a solver, compare it with finite differences of the flux function on a few values:

```{literalinclude} snippet/fv_scheme/fv_burgers.cpp
  :language: c++
  :start-at: // Largest gap between the Jacobian
  :end-before: // Largest |f - g|
```

The program checks the scheme against the exact $\nabla \cdot f(u) = u\,(\partial_x u + \partial_y u)$ for $u = \sin(2\pi x)\sin(2\pi y)$, on a mesh periodic in both directions, then checks the Jacobian:

```bash
./fv_burgers
```

```{literalinclude} snippet/fv_scheme/fv_burgers_output.txt
  :language: text
```

The error decreases at order 2 and the Jacobian matches the finite differences to $10^{-11}$, the accuracy of a centered difference with a step of $10^{-6}$.
The column `pair` is explained in the next section.
In an implicit solve, a correct Jacobian also shows as a quadratic decrease of the Newton residual, as the [PETSc how-to guide](petsc.md#solve-a-non-linear-scheme) shows.

## Write a non-conservative flux

A conservative scheme gives the cell on the right of a face the opposite of the flux it gives the cell on the left.
To give them two independent contributions, set `flux_function` instead of `cons_flux_function`.
It fills a `samurai::FluxValuePair<cfg>`: `flux[0]` is added to the cell on the left and `flux[1]` to the cell on the right.
Non-conservative fluxes exist only for `SchemeType::NonLinear`.

The example writes the conservative Burgers flux in this form, which is a way to check the sign convention before you write a flux that is really non-conservative:

```{literalinclude} snippet/fv_scheme/fv_burgers.cpp
  :language: c++
  :start-at: // The same flux, written as a pair
  :end-before: // Largest gap
```

The column `pair` of the output above is the largest difference between the two schemes, 0 at every level.
For an implicit solve, the Jacobian counterpart is `jacobian_function`, which fills a `samurai::StencilJacobianPair<cfg>`.

## Use a wider stencil

Set the stencil of each direction when the flux reads more than the two cells around the face.
The stencil is a list of direction vectors from the cell on the left of the face, which is `{0, 0}` in 2D.
`samurai::line_stencil<dim, d>(-1, 0, 1, 2)` builds the stencil of four cells along the direction `d`, and you assign it to `my_flux[d].stencil`.
The values reach the flux function in the order of the stencil: here `u[0]` is the second cell on the left of the face, `u[1]` and `u[2]` the two cells next to the face, and `u[3]` the second cell on the right.
The {ref}`stencil configuration <stencil-configuration>` section of the reference describes the other helpers.

The mesh must provide as many ghosts as the stencil reaches: set `max_stencil_radius` in the mesh configuration to half the stencil size, 2 for a stencil of size 4 (the default) and 3 for size 6.
A scheme whose stencil does not fit throws an error when you apply it, as the [troubleshooting guide](troubleshooting.md#scheme-stencil-larger-than-the-mesh-allows) explains.

If the cells the flux reads depend on the solution, as in an upwind or WENO scheme, configure a stencil that holds every cell the flux may read and choose among them in the flux function.

## Solve the scheme implicitly

Pass the scheme, or a combination such as `id + dt * my_scheme`, to `samurai::petsc::make_solver`, as the [PETSc how-to guide](petsc.md) explains.
The unknown field needs `samurai::Dirichlet` or `samurai::Neumann` conditions: the matrix assembly writes the equations of these two conditions only.
A `SchemeType::NonLinear` scheme also needs its Jacobian function, or the option `-snes_mf`.

## Related

- The {doc}`finite volume schemes reference <../reference/finite_volume_schemes>`, for the finite volume principle, the full description of `FluxConfig`, `FluxDefinition`, `SchemeType` and the flux function types, and the list of built-in schemes.
- The {doc}`local schemes reference <../reference/local_schemes>`, for schemes computed cell by cell rather than through fluxes.
- {doc}`bc`, for the boundary conditions that fill the ghosts a scheme reads.
- {doc}`petsc`, to solve an implicit scheme.
- The {doc}`reaction-diffusion tutorial <../tutorial/reaction_diffusion>` and the [flux mechanism chapter](https://hpc-maths.github.io/2025-hands-on-samurai/flux-burgers) of the hands-on course, which build schemes inside a complete simulation.
