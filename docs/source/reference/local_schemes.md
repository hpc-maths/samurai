# Local schemes

A local scheme is a discrete operator whose value in a cell depends only on the input field in that cell, such as a reaction term or a change of variables.
{{ project }} implements it as a *cell-based scheme*: a scheme evaluated cell by cell, with no flux between cells.
A cell-based scheme can also read a star stencil around the cell, as described in [Stencil-based cell schemes](#stencil-based-cell-schemes).
Schemes built from the fluxes between neighbouring cells are described in the {doc}`finite volume schemes reference <finite_volume_schemes>`,
and the {doc}`reaction-diffusion tutorial <../tutorial/reaction_diffusion>` uses a non-linear local scheme in a complete program.

The cell-based schemes are part of the finite volume module:

```c++
#include <samurai/schemes/fv.hpp>
```

The examples on this page use a 1D mesh of four cells of length $h = 1/4$ on $[0, 1)$ and a field `u` equal to the cell center:

```c++
static constexpr std::size_t dim = 1;
samurai::Box<double, dim> box({0.}, {1.});
auto config = samurai::mesh_config<dim>().min_level(2).max_level(2);
auto mesh   = samurai::mra::make_mesh(box, config);

auto u = samurai::make_scalar_field<double>("u", mesh);
samurai::for_each_cell(mesh,
                       [&](const auto& cell)
                       {
                           u[cell] = cell.center(0);
                       });
```

## Implementing a local scheme

Consider a local discrete operator $\mathcal{A}$ applied to a field $u$, and let $v$ denote the resulting field:

$$
v = \mathcal{A}(u).
$$

### Static configuration

The structure of the scheme is declared first, in a `samurai::LocalCellSchemeConfig` type.
Its template parameters are, in this order:

1. `scheme_type`: one of the values of `samurai::SchemeType`:

   ```c++
   enum class SchemeType
   {
       NonLinear,
       LinearHeterogeneous,
       LinearHomogeneous
   };
   ```

   Cell-based schemes accept `SchemeType::NonLinear` and `SchemeType::LinearHomogeneous`.
   `SchemeType::LinearHeterogeneous` has no cell-based implementation: `make_cell_based_scheme` fails to compile with it (see [Errors](#errors)).
2. `output_field_type`: the C++ type of the resulting field $v$.
3. `input_field_type`: the C++ type of the field $u$.

```c++
using cfg = samurai::LocalCellSchemeConfig<
    samurai::SchemeType::NonLinear, // scheme_type
    decltype(u),                    // output_field_type (same as input)
    decltype(u)>;                   // input_field_type
```

The output and input field types may differ, for instance to map a scalar field to a vector field.

### Discrete operator creation and usage

The discrete operator is created from `cfg`, with an optional name:

```c++
auto A = samurai::make_cell_based_scheme<cfg>("A");
```

Without a name, the discrete operator is named "(unnamed)"; `A.set_name("...")` sets it later.
The name appears in error messages, in timers and in the name of the output field.
The function that defines $\mathcal{A}$ depends on the scheme type:

| Scheme type | Definition | Section |
| --- | --- | --- |
| `SchemeType::LinearHomogeneous` | `A.set_coefficients_func(...)` | [Linear, homogeneous local schemes](#linear-homogeneous-local-schemes) |
| `SchemeType::NonLinear` | `A.set_scheme_function(...)`, and `A.set_jacobian_function(...)` if `A` is solved implicitly | [Non-linear local schemes](#non-linear-local-schemes) |

Once defined, the discrete operator is used in an explicit context

```c++
auto v = A(u); // a new field, named "A(u)"
```

or in an implicit context

```c++
samurai::petsc::solve(A, x, b); // solves the equation A(x) = b
```

`samurai::petsc::solve` requires `SAMURAI_WITH_PETSC` and picks the solver from the scheme type:

- `SchemeType::LinearHomogeneous`: a linear solver, which assembles the matrix of $\mathcal{A}$;
- `SchemeType::NonLinear`: one independent Newton solve per cell, of size the number of field components.
  The initial guess is the value of `x` in the cell, and each cell solve uses LU by default.
  The PETSc options of the command line (`-snes_monitor`, `-snes_rtol`...) apply to every cell solve.

A local scheme reads no ghost, so neither its explicit application nor its implicit solve needs boundary conditions on the input field.

## Linear, homogeneous local schemes

A linear, homogeneous local scheme is given by a matrix $C$ that does not depend on the cell:

$$
v_{|V} = C\, u_{|V} \qquad \text{for every cell } V.
$$

Its coefficient function fills $C$ and has the signature

```c++
void(samurai::StencilCoeffs<cfg>& coeffs, double h)
```

where `h` is the cell length.
The coefficients are computed once per mesh level and reused for all the cells of that level.
They may depend on `h` and on parameters captured by the lambda function, but not on cell values.
`set_coefficients_func` sets the coefficient function; `coefficients_func()` returns a reference to it, which can also be assigned.

For a local scheme, `samurai::StencilCoeffs<cfg>` is a matrix of size `output_n_comp x input_n_comp`,
where `output_n_comp` (resp. `input_n_comp`) is the number of components of `output_field_type` (resp. `input_field_type`).
It is an `xtensor` matrix, whose coefficient at indices $(k, l)$ is `coeffs(k, l)`.
When both fields are scalar fields, it reduces to a scalar type (typically `double`).

The scalar discrete operator $\mathcal{L}(u) = a u$ reads

```c++
double a = 3;

using cfg = samurai::LocalCellSchemeConfig<
    samurai::SchemeType::LinearHomogeneous, decltype(u), decltype(u)>;

auto L = samurai::make_cell_based_scheme<cfg>("L");

L.set_coefficients_func(
    [a](samurai::StencilCoeffs<cfg>& coeffs, double /* h */)
    {
        coeffs = a; // L(u) = a * u
    });

auto v = L(u);
```

and gives, in the four cells,

```text
u = 0.125, L(u) = 0.375
u = 0.375, L(u) = 1.125
u = 0.625, L(u) = 1.875
u = 0.875, L(u) = 2.625
```

`samurai::petsc::solve(L, x, v)`, with `x` initialized to 0, returns `x` equal to `u`.

With a scalar input field and a 2-component output field, `coeffs` is a $2 \times 1$ matrix:

```c++
auto w = samurai::make_vector_field<double, 2>("w", mesh);

using cfg2 = samurai::LocalCellSchemeConfig<
    samurai::SchemeType::LinearHomogeneous, decltype(w), decltype(u)>;

auto B = samurai::make_cell_based_scheme<cfg2>("B");

B.set_coefficients_func(
    [](samurai::StencilCoeffs<cfg2>& coeffs, double)
    {
        coeffs(0, 0) = 1;
        coeffs(1, 0) = -2;
    });

auto Bu = B(u); // 2-component field
```

```text
B(u) = { 0.125, -0.25 }
B(u) = { 0.375, -0.75 }
B(u) = { 0.625, -1.25 }
B(u) = { 0.875, -1.75 }
```

The identity, zero and buoyancy discrete operators of the finite volume module are linear, homogeneous local schemes (see {doc}`finite_volume_schemes`).

## Non-linear local schemes

The formula of a non-linear local scheme is a lambda function, given to `set_scheme_function`:

```c++
double k = 10;

using cfg = samurai::LocalCellSchemeConfig<
    samurai::SchemeType::NonLinear, // scheme_type
    decltype(u),                    // output_field_type
    decltype(u)>;                   // input_field_type

auto R = samurai::make_cell_based_scheme<cfg>("reaction");

R.set_scheme_function(
    [&](samurai::SchemeValue<cfg>& value,
        const auto& cell,
        const auto& field)
    {
        auto v = field[cell];
        value  = k * v * v * (1 - v);
    });
```

The parameters of the function are:

- `value`: the value of the discrete operator in the cell, to fill;
- `cell`: the current cell;
- `field`: the input field.

`samurai::SchemeValue<cfg>` is an array of size `output_field_type::n_comp`, based on `xtensor`: its $i$-th component is `value(i)`.
If `output_field_type` is a scalar field, it reduces to a scalar type (typically `double`).

`field` must be declared `const auto&`.
`set_scheme_function` stores the function twice:
once for the input field type, used in explicit applications,
and once for `samurai::LocalField<input_field_type>`, a view on the value of a single cell, used by the cell-by-cell implicit solve.
A function declared with the input field type does not convert to the second form and fails to compile (see [Errors](#errors)).

`R(u)` gives, in the four cells,

```text
u = 0.125, R(u) = 0.136719
u = 0.375, R(u) = 0.878906
u = 0.625, R(u) = 1.46484
u = 0.875, R(u) = 0.957031
```

### Jacobian function

An implicit solve needs the Jacobian matrix of the scheme with respect to the cell value.
It is given by `set_jacobian_function`; an explicit application does not use it.

```c++
R.set_jacobian_function(
    [&](samurai::JacobianMatrix<cfg>& jac,
        const auto& cell,
        const auto& field)
    {
        auto v = field[cell];
        jac    = k * (2 * v * (1 - v) - v * v);
    });
```

`samurai::JacobianMatrix<cfg>` is a matrix of size `output_n_comp x input_n_comp`: `jac(k, l)` is the derivative of `value(k)` with respect to the $l$-th component of the input field.
If both `output_field_type` and `input_field_type` are scalar fields, it reduces to a scalar type (typically `double`).
As for the scheme function, `field` must be declared `const auto&`.

```{important}
The implicit solve of a non-linear local scheme does not compile when the unknown is a scalar field:
the cell-by-cell solver reads the Jacobian as a matrix, `jac(i, j)`, which a `double` does not provide.
Use a vector field with one component instead, `samurai::make_vector_field<double, 1>`, as the Nagumo demo does.
```

### Vector field example

For a 2-component field `u`, the discrete operator $\mathcal{A}(u) = (u_0 u_1,\; u_0 + u_1^2)$ and its Jacobian are

```c++
using cfg = samurai::LocalCellSchemeConfig<
    samurai::SchemeType::NonLinear, decltype(u), decltype(u)>;

auto A = samurai::make_cell_based_scheme<cfg>("A");

A.set_scheme_function(
    [](samurai::SchemeValue<cfg>& value,
       const auto& cell,
       const auto& field)
    {
        auto v   = field[cell];
        value(0) = v(0) * v(1);
        value(1) = v(0) + v(1) * v(1);
    });

A.set_jacobian_function(
    [](samurai::JacobianMatrix<cfg>& jac,
       const auto& cell,
       const auto& field)
    {
        auto v    = field[cell];
        jac(0, 0) = v(1); // d(value(0)) / d(v(0))
        jac(0, 1) = v(0); // d(value(0)) / d(v(1))
        jac(1, 0) = 1;
        jac(1, 1) = 2 * v(1);
    });
```

With $u = (x, 1 + x)$ in each cell of center $x$, `auto b = A(u)` gives

```text
A(u) = { 0.140625,  1.390625}
A(u) = { 0.515625,  2.265625}
A(u) = { 1.015625,  3.265625}
A(u) = { 1.640625,  4.390625}
```

and `samurai::petsc::solve(A, x, b)`, from the initial guess $x = (1, 1)$, recovers $u$:

```text
x = { 0.125   ,  1.125   }
x = { 0.375   ,  1.375   }
x = { 0.625   ,  1.625   }
x = { 0.875   ,  1.875   }
```

## Stencil-based cell schemes

A cell-based scheme can also read the cells of a star stencil around the cell it writes:
the cell itself and, in each Cartesian direction, its neighbours up to a given stencil radius.
The configuration type is `samurai::StarStencilSchemeConfig`, whose template parameters are the scheme type, the stencil radius (1 or 2), the output field type and the input field type.
`samurai::LocalCellSchemeConfig` is the same type with a stencil radius of 0.

```c++
using cfg = samurai::StarStencilSchemeConfig<
    samurai::SchemeType::LinearHomogeneous, 1, decltype(u), decltype(u)>;
```

The configuration exposes `cfg::stencil_size`, equal to $1 + 2\,\text{dim}\times\text{radius}$, and `cfg::center_index`, the position of the cell itself in the stencil.
The cells of the stencil come in this order:

| Dimension | Radius 1 | Radius 2 |
| --- | --- | --- |
| 1 | left, center, right | left2, left, center, right, right2 |
| 2 | left, center, right, bottom, top | left2, left, center, right, right2, bottom2, bottom, top, top2 |
| 3 | left, center, right, front, back, bottom, top | left2, left, center, right, right2, front2, front, back, back2, bottom2, bottom, top, top2 |

`center_index` equals the radius.
The scheme reads the ghosts of the input field, which must therefore have boundary conditions.
The explicit application updates these ghosts first, if they are out of date.

### Linear, homogeneous

The coefficient function has the same signature as for a local scheme,
but `samurai::StencilCoeffs<cfg>` is now an array of `stencil_size` matrices: `coeffs[i]` multiplies the value of the $i$-th stencil cell.
The 1D discrete operator $-u''$, with Dirichlet conditions on `u`, reads

```c++
samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

// Star stencil of radius 1: in 1D, {left, center, right}
using cfg = samurai::StarStencilSchemeConfig<
    samurai::SchemeType::LinearHomogeneous, 1, decltype(u), decltype(u)>;

auto D = samurai::make_cell_based_scheme<cfg>("D");

D.set_coefficients_func(
    [](samurai::StencilCoeffs<cfg>& coeffs, double h)
    {
        coeffs[0] = -1 / (h * h); // left
        coeffs[1] = 2 / (h * h);  // center
        coeffs[2] = -1 / (h * h); // right
    });
```

`samurai::make_diffusion_cell_based` implements this discrete operator in any dimension.

### Non-linear

The scheme function receives the array of the stencil cells, `samurai::StencilCells<cfg>`, in place of the single cell.
The Jacobian function fills `samurai::StencilJacobian<cfg>`, an array of `stencil_size` matrices: `jac[i]` is the derivative with respect to the value of the $i$-th stencil cell.
The non-linear version of `D` reads

```c++
using nl_cfg = samurai::StarStencilSchemeConfig<
    samurai::SchemeType::NonLinear, 1, decltype(u), decltype(u)>;

auto N = samurai::make_cell_based_scheme<nl_cfg>("N");

N.set_scheme_function(
    [](samurai::SchemeValue<nl_cfg>& value,
       const samurai::StencilCells<nl_cfg>& cells,
       const decltype(u)& field)
    {
        static constexpr std::size_t left   = 0;
        static constexpr std::size_t center = nl_cfg::center_index;
        static constexpr std::size_t right  = 2;

        double h = cells[center].length;
        value    = (2 * field[cells[center]] - field[cells[left]]
                    - field[cells[right]]) / (h * h);
    });

N.set_jacobian_function(
    [](samurai::StencilJacobian<nl_cfg>& jac,
       const samurai::StencilCells<nl_cfg>& cells,
       const auto& /* field */)
    {
        double h = cells[nl_cfg::center_index].length;
        jac[0]   = -1 / (h * h); // derivative w.r.t. the left value
        jac[1]   = 2 / (h * h);  // derivative w.r.t. the center value
        jac[2]   = -1 / (h * h); // derivative w.r.t. the right value
    });
```

The scheme function may declare `field` with the input field type, because the cell-by-cell solve applies only to local schemes.
The Jacobian function may not: `set_jacobian_function` always stores the `samurai::LocalField` form too, so `field` must be declared `const auto&`.

With $u = x^2$, `D(u)` and `N(u)` give the same values:

```text
u = 0.015625, D(u) = -1.5, N(u) = -1.5
u = 0.140625, D(u) = -2, N(u) = -2
u = 0.390625, D(u) = -2, N(u) = -2
u = 0.765625, D(u) = 30.5, N(u) = 30.5
```

The two interior cells give the exact value $-2$; the boundary cells read the Dirichlet ghosts.
`samurai::petsc::solve` on `D` uses a linear solver, and on `N` a Newton solver over the whole mesh, as for a flux-based scheme;
the unknown needs boundary conditions in both cases.

The level-set reinitialization of the {doc}`level-set tutorial <../tutorial/level_set>` is a non-linear cell-based scheme on a star stencil of radius 2.

## Combining cell-based schemes

Cell-based schemes combine with the multiplication by a scalar, `+` and `-`:

| Expression | Result |
| --- | --- |
| `s * A`, with `s` a `double` | a cell-based scheme of the same type, named `"s * A"` |
| `-A` | `(-1) * A` |
| `L + R`, `L - R`, with `L` linear homogeneous and `R` non-linear, both local | one non-linear local scheme, with a Jacobian function if `R` has one |
| two cell-based schemes of different configurations, otherwise | a `samurai::OperatorSum` |
| a cell-based scheme and a flux-based scheme | a `samurai::OperatorSum` |
| two cell-based schemes of the same configuration, such as `L + L` or `R - R` | does not compile |

A `samurai::OperatorSum` applies its discrete operators one after the other, and adds their contributions to the same matrix in an implicit context (see {doc}`finite_volume_schemes`).
With the identity `id = samurai::make_identity<decltype(u)>()`, the reaction $\mathcal{R}(u) = u^2$ and `dt = 0.5`, the explicit applications give

```text
2*R [2 * R]: 0.03125 0.28125 0.78125 1.53125
id + R [Identity + R]: 0.140625 0.515625 1.01562 1.64062
id - dt*R [Identity + -1 * 5.0e-01 * R]: 0.117188 0.304688 0.429688 0.492188
R + id [[ R + Identity ]]: 0.140625 0.515625 1.01562 1.64062
```

where the bracket holds the name of the discrete operator.
`id + R` is a single cell-based scheme, while `R + id` is a `samurai::OperatorSum`, whose name lists its terms between square brackets.
The order matters for the implicit solve: `id - dt * R` is solved cell by cell, as in the {doc}`reaction-diffusion tutorial <../tutorial/reaction_diffusion>`.

## Errors

`make_cell_based_scheme` with `SchemeType::LinearHeterogeneous` fails at compile time:

```text
static assertion failed: Either the required file has not been included, or the CellBasedScheme class has not been specialized for this type of scheme.
```

A scheme or Jacobian function of a local scheme whose `field` parameter is not `const auto&` fails at compile time in `set_scheme_function` or `set_jacobian_function`, with `no match for 'operator='` on a `std::function` that takes a `samurai::LocalField`.

A non-linear local scheme without Jacobian function is solved implicitly (`std::runtime_error`):

```text
The function 'local_jacobian_function' of operator 'reaction' has not been implemented.
```

A non-linear local scheme whose function was assigned with `scheme_function() = ...` instead of `set_scheme_function` is solved implicitly (`std::runtime_error`):

```text
The scheme function 'local_scheme_function' of operator 'reaction' has not been implemented.
```

Assigning `scheme_function()` stores only the form for the input field type; `set_scheme_function` stores both.

A non-linear stencil-based scheme without Jacobian function is solved implicitly (`std::runtime_error`, raised when the Jacobian matrix is assembled):

```text
The jacobian function of operator 'N' has not been implemented.
Use option -snes_mf or -snes_fd for an automatic computation of the jacobian matrix.
```
