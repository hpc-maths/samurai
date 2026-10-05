# Finite volume schemes

The finite volume module of {{ project }} builds finite volume schemes on adapted meshes.
It provides ready-made discrete operators (diffusion, convection, gradient, divergence and others, listed in [Available implementations](#available-implementations)) and lets you define your own flux functions.
To write your own scheme step by step, follow the {doc}`finite volume scheme how-to guide <../howto/fv_scheme>`.

To use it, include

```c++
#include <samurai/schemes/fv.hpp>
```

The PETSc-based solvers (`samurai::petsc::solve`, `samurai::petsc::make_solver`) are available when the program is compiled with `SAMURAI_WITH_PETSC` defined.

(fv-principle)=

## Finite volume principle

In the finite volume method (FVM), the computed values are average values of the physical variables over control volumes.
The system of equations is therefore integrated over each control volume $V$.
Let $u$ denote a conservative field, and $\mathcal{D}(u)$ a differential term to integrate over $V$.
To ensure flux conservation, the integral over $V$ is rewritten as a sum of fluxes through the faces of $V$:

$$
\int_V \mathcal{D}(u) = \int_{\partial V} \mathcal{F}(u),
$$

where $\mathcal{F}$ is the flux function associated with the differential term $\mathcal{D}$.
For each differential term, the expression of $\mathcal{F}$ follows from Green's theorem or one of its corollaries.
For instance, if $\mathcal{D}(u) = \Delta u$, then $\mathcal{F}(u) = \nabla u\cdot \mathbf{n}$,
where $\mathbf{n}$ is the unit vector normal to $\partial V$ pointing out of $V$.
In the discrete setting, $\partial V$ is a set of faces,
and on each face $F$, the flux $\mathcal{F}(u)$ is approximated by a discrete counterpart $\mathcal{F}_h(u_h)$.
This approximation is constant on each face, so that the integral becomes

$$
\int_{\partial V} \mathcal{F}(u) = \sum_{F\subset \partial V}\int_F \mathcal{F}(u) \approx \sum_{F\subset \partial V} |F| \mathcal{F}_h(u_h)_{|F}.
$$

From a {{ project }} mesh and an implementation of $\mathcal{F}_h(u_h)$, the finite volume module builds a discrete differential operator
that can be called like a function in an explicit context, or passed to a solver in an implicit context.
When it runs, it iterates over all cell interfaces $F$ and computes the flux $\mathcal{F}_h(u_h)_{|F}$ through each of them.

```{figure} ./figures/flux.svg
:width: 25%
:align: center
```

Let $V_L$ and $V_R$ be the two cells sharing the face $F$, ordered along the corresponding Cartesian direction
(in the x-direction, $V_L$ and $V_R$ are the left and right cells).
The contributions of the face to these cells are

$$
\begin{aligned}
\mathcal{C}_L &:= \;\;\; \frac{|F|}{|V_L|} \mathcal{F}_h(u_h)_{|F}, \\
\mathcal{C}_R &:=      - \frac{|F|}{|V_R|} \mathcal{F}_h(u_h)_{|F}.
\end{aligned}
$$

The flux $\mathcal{F}_h(u_h)_{|F}$ is computed once and used for both contributions.

```{note}
Flux conservation requires the flux in one direction to be the opposite of the flux in the other direction.
{{ project }} also accepts two fluxes $\mathcal{F}_h^+(u_h)_{|F}$ and $\mathcal{F}_h^-(u_h)_{|F}$ with $\mathcal{F}_h^-(u_h)_{|F} \neq -\mathcal{F}_h^+(u_h)_{|F}$:
see {ref}`non-conservative schemes <non_conservative_schemes>`.
```

Where a level jump occurs, ghosts take part in the computation of $\mathcal{F}_h(u_h)_{|F}$, so that the flux is always computed between two cells of the same length.

```{figure} ./figures/flux_level_jump.svg
:width: 25%
:align: center
```

$V_L$ and $V_R$ still denote real cells in the formulas of $\mathcal{C}_L$ and $\mathcal{C}_R$.

```{note}
The contributions are divided by the measure of the corresponding cell.
Instead of $\int_V \mathcal{D}(u)$, the discrete operator computes $\frac{1}{|V|} \int_V \mathcal{D}(u)$,
so that discrete source terms do not have to be multiplied by the cell measures.
```

## Flux-based schemes

A flux-based scheme is described by a scheme type (`samurai::SchemeType`), a static configuration (`samurai::FluxConfig`) and the flux through the faces of each Cartesian direction (`samurai::FluxDefinition`).
`samurai::make_flux_based_scheme` turns the flux definition into a discrete operator.
The {doc}`finite volume scheme how-to guide <../howto/fv_scheme>` writes such schemes step by step and checks them against the built-in ones.

### samurai::SchemeType

```c++
enum class SchemeType
{
    NonLinear,
    LinearHeterogeneous,
    LinearHomogeneous
};
```

| Value | Flux | Flux function returns | Implicit solver |
| --- | --- | --- | --- |
| `LinearHomogeneous` | linear in the field, coefficients that depend on the cell length only | the coefficients of the stencil values | linear (KSP) |
| `LinearHeterogeneous` | linear in the field, coefficients that vary in space | the coefficients of the stencil values | linear (KSP) |
| `NonLinear` | any function of the stencil values | the value of the flux | Newton (SNES) |

The coefficients of a `LinearHomogeneous` scheme are computed once per mesh level and reused for every face of that level.
The coefficients of a `LinearHeterogeneous` scheme are computed for each face.

### samurai::FluxConfig

```c++
template <SchemeType scheme_type,
          std::size_t stencil_size,
          class output_field_type,
          class input_field_type,
          class parameter_field_type = void*>
struct FluxConfig;
```

| Template parameter | Meaning |
| --- | --- |
| `scheme_type` | a value of `samurai::SchemeType` |
| `stencil_size` | the number of cells the flux of one face reads, 2 for a flux between the two cells of the face |
| `output_field_type` | the type of the field the scheme returns |
| `input_field_type` | the type of the field the scheme applies to |
| `parameter_field_type` | the type of a field the flux function reads in addition to the input field, such as a velocity field; `void*` for none. See [Parameter fields](#parameter-fields). |

`FluxConfig` exposes these parameters as `scheme_type`, `stencil_size`, `output_field_t`, `input_field_t` and `parameter_field_t`, plus `has_parameter_field` and the space dimension `dim`.
The output and input fields can have different numbers of components: the gradient of a scalar field has as many components as the space dimension.

The stencil size bounds the stencil radius the mesh must provide ghosts for.
A discrete operator whose `stencil_size` is larger than `max_stencil_size()` of the mesh configuration fails when it is applied (see [Errors](#errors)).

(stencil-configuration)=

### Stencils

The flux in direction `d` is computed between one cell and its neighbour along the Cartesian unit vector of `d`: in 2D, from left to right and from bottom to top.
The stencil is an array of direction vectors from the origin cell.
The origin cell is the left cell of the face in the horizontal direction and the bottom cell in the vertical direction, and is represented by the zero vector `{0,0}` (or `{0,0,0}` in 3D).
For instance, in the x-direction, the stencil `{{0,0}, {1,0}}` captures the origin cell and its right neighbour.

```{figure} ./figures/flux_stencils.svg
:width: 60%
:align: center
```

In this figure, the flux at the red interface reads four cells, and the blue arrow shows its orientation.
Each cell is labelled with the direction vector that captures it: the stencils are `{{-1,0}, {0,0}, {1,0}, {2,0}}` in the x-direction and `{{0,-1}, {0,0}, {0,1}, {0,2}}` in the y-direction.
The flux function receives the cells and the values of the stencil in the order of the configured stencil.

Two helpers build line stencils in any direction:

| Helper | Stencil |
| --- | --- |
| `samurai::line_stencil<dim, d>(i_0, i_1, ...)` | the cells at offsets `i_0, i_1, ...` from the origin cell along direction `d` |
| `samurai::line_stencil_from<dim, d, size>(i_0)` | `size` consecutive cells along direction `d`, from offset `i_0` |

`samurai::line_stencil<dim, 0>(-1, 0, 1, 2)` and `samurai::line_stencil_from<dim, 0, 4>(-1)` both build the x-direction stencil of the figure.

If the stencil is not set, `FluxDefinition` uses `samurai::line_stencil_from<dim, d, stencil_size>(-stencil_size / 2 + 1)`.
If `stencil_size` is even, the cells are evenly distributed on both sides of the face; if it is odd, there is one more cell on the positive side.

If the cells a flux reads depend on a value known only at run time (for instance the sign of the velocity in the upwind or WENO schemes), the stencil holds every cell the flux may read and the flux function selects the cells it uses.

### samurai::FluxDefinition

`samurai::FluxDefinition<cfg>` holds one normal flux definition per positive Cartesian direction; `flux[d]` is the definition in direction `d`.
Each one has the members:

| Member | Scheme types | Content |
| --- | --- | --- |
| `direction` | all | the unit vector of direction `d`, set by the constructor |
| `stencil` | all | the stencil of the flux, see [Stencils](#stencils) |
| `cons_flux_function` | all | the conservative flux function |
| `cons_jacobian_function` | `NonLinear` | the derivative of the conservative flux with respect to the stencil values |
| `flux_function` | `NonLinear` | the non-conservative flux function, see [Non-conservative fluxes](#non-conservative-fluxes) |
| `jacobian_function` | `NonLinear` | the derivative of the non-conservative flux |

The default constructor sets the default stencil and no function.
`FluxDefinition<cfg>(f)` also sets `f` as `cons_flux_function` in every direction.

`samurai::make_flux_based_scheme` copies the flux definition into the discrete operator, so the discrete operator can outlive the `FluxDefinition` object.
The functions are copied too, with what they capture: a function that captures a local variable by reference must not outlive that variable.

(lin_homog_operators)=

### Flux functions

The signature of `cons_flux_function` depends on the scheme type:

| Scheme type | Signature |
| --- | --- |
| `LinearHomogeneous` | `void(samurai::FluxStencilCoeffs<cfg>& c, double h)` |
| `LinearHeterogeneous` | `void(samurai::FluxStencilCoeffs<cfg>& c, const samurai::StencilData<cfg>& data)` |
| `NonLinear` | `void(samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& u)` |

#### Linear schemes

A linear flux function does not compute the flux itself: it returns the coefficients $(c_i)_i$ of the linear combination

```{math}
:label: linear_comb

\mathcal{F}_h(u_h)_{|F} := \sum_i c_i u_i, \qquad \text{ where } u_i := (u_h)_{|V_i},
```

where $(V_i)_i$ are the cells of the stencil.
In an explicit context they are the coefficients of this combination; in an implicit context they are inserted into the matrix of the linear system.

`samurai::FluxStencilCoeffs<cfg>` is a fixed-size array with one coefficient per stencil cell.
Each `c[i]` is an `xtensor` matrix of size `output_n_comp x input_n_comp`, the numbers of components of the output and input fields: `c[i](k, l)` is the coefficient of component `l` of the input in component `k` of the flux.
When both fields are scalar fields, `c[i]` reduces to a scalar (typically `double`), with no accessor and no `xtensor` function.

A `LinearHomogeneous` flux function receives the cell length `h`, the only quantity its coefficients may depend on.
Other constant parameters are captured by the function.

(lin_heter_operators)=

#### Linear heterogeneous schemes

A `LinearHeterogeneous` flux function receives a `samurai::StencilData<cfg>` instead of the cell length:

| Member of `StencilData` | Content |
| --- | --- |
| `cells` | the cells of the computational stencil, in the order of the configured stencil |
| `cell_length` | the length of these cells |

The flux function reads its parameters in these cells.

````{warning}
The cells are those of the *computational* stencil, not the two real cells around the face.
Where a level jump occurs, at least one of the computational cells is a ghost,
so the parameters must hold values in the ghosts.
A material parameter can be set on all cells, ghosts included, with a loop over the `reference` cells:

```c++
using mesh_id_t = decltype(mesh)::mesh_id_t;

auto set_param = [&](auto& cell)
{
    param[cell] = cell.center(0) < 0 ? 1. : 10.;
};
samurai::for_each_cell(mesh[mesh_id_t::reference], set_param);
```

For a computed field used as a parameter (for instance, a velocity field computed from the Navier-Stokes equations),
update its ghosts with `samurai::update_ghost_mr(param);` (or `samurai::update_ghost(param);` on an AMR mesh),
or declare it as a [parameter field](#parameter-fields) so that the discrete operator updates them.
````

#### Non-linear schemes

A `NonLinear` flux function computes the flux from the stencil values:

| Type | Content |
| --- | --- |
| `samurai::FluxValue<cfg>` | the flux, an array of `output_n_comp` values, or a scalar if the output field is a scalar field |
| `samurai::StencilData<cfg>` | the cells of the computational stencil and their length, as for the heterogeneous schemes |
| `samurai::StencilValues<cfg>` | the values of the input field in the stencil cells, in the order of the configured stencil |

### Jacobian function

An implicit non-linear scheme is solved by a Newton method, which needs the Jacobian matrix of the flux with respect to the stencil values.
It is given by `cons_jacobian_function`, next to `cons_flux_function`.
Its signature is

```c++
void(samurai::StencilJacobian<cfg>& jac,
     const samurai::StencilData<cfg>& data,
     const samurai::StencilValues<cfg>& field)
```

`jac[i]` is the derivative of the flux with respect to the values of the `i`-th stencil cell:
a matrix of size `output_n_comp x input_n_comp`, which reduces to a scalar when both fields are scalar fields.
Without a Jacobian function, the explicit application works, but the implicit assembly fails (see [Errors](#errors)).
The PETSc option `-snes_mf` (matrix-free Jacobian) solves without it. `-snes_fd` (finite-difference Jacobian) does not work with the default direct solver: the residual does not depend on the ghost unknowns, so the matrix has zero rows there and the factorization fails with a zero pivot ([#625](https://github.com/hpc-maths/samurai/issues/625)).

(parameter_field)=

### Parameter fields

The fifth template parameter of `FluxConfig` declares the type of a parameter field.
After the discrete operator is created, `set_parameter_field` registers the field itself:

```c++
auto conv = samurai::make_flux_based_scheme(upwind);
conv.set_parameter_field(a); // conv keeps the ghosts of a up to date
```

The discrete operator then updates the ghosts of the parameter field, if they are out of date, before it applies the scheme explicitly,
and before the implicit assembly of a non-linear or heterogeneous scheme.
The update fills the ghosts from the cell values and applies the boundary conditions attached to the parameter field.
A discrete operator has at most one parameter field.

(non_conservative_schemes)=

### Non-conservative fluxes

Two-phase flow simulations give examples of non-conservative schemes: the scheme is conservative within each phase, but non-conservative fluxes can be computed at the interface between phases.
The contributions of a face $F$ to its cells $V_L$ and $V_R$ are then

$$
\begin{aligned}
\mathcal{C}_L &:= \frac{|F|}{|V_L|} \mathcal{F}_h^+(u_h)_{|F}, \\
\mathcal{C}_R &:= \frac{|F|}{|V_R|} \mathcal{F}_h^-(u_h)_{|F},
\end{aligned}
$$

where $\mathcal{F}_h^-(u_h)_{|F} = -\mathcal{F}_h^+(u_h)_{|F}$ does not necessarily hold.

A conservative scheme sets $\mathcal{F}_h(u_h)_{|F}$ through `cons_flux_function`.
A non-conservative scheme sets both $\mathcal{F}_h^+(u_h)_{|F}$ and $\mathcal{F}_h^-(u_h)_{|F}$ through `flux_function`, whose signature is that of `cons_flux_function` with a `samurai::FluxValuePair<cfg>` instead of a `samurai::FluxValue<cfg>`:
`flux[0]` is $\mathcal{F}_h^+(u_h)_{|F}$, added to $V_L$, and `flux[1]` is $\mathcal{F}_h^-(u_h)_{|F}$, added to $V_R$.
Non-conservative fluxes are available only with `SchemeType::NonLinear`: the flux definitions of the linear scheme types have no `flux_function` member.

If `flux_function` is not set, it is derived from `cons_flux_function`, with `flux[1] = -flux[0]`.
For implicit schemes, the Jacobian counterpart is `jacobian_function`, which fills a `samurai::StencilJacobianPair<cfg>`;
if it is not set, it is derived from `cons_jacobian_function`.

### Discrete operator creation and usage

`samurai::make_flux_based_scheme(flux_definition)` creates the discrete operator.
`samurai::make_divergence(flux_definition)` is an alias, provided for readability when the scheme is the divergence of a flux, since

$$
\int_V \nabla\cdot f(u) = \int_{\partial V} f(u) \cdot \mathbf{n}.
$$

The discrete operator `D` applies in an explicit context

```c++
auto v = D(u);
```

or in an implicit context

```c++
// Right-hand side
auto rhs = samurai::make_scalar_field<double>("rhs", mesh, 0.);
// Solves the equation D(u) = rhs
samurai::petsc::solve(D, u, rhs);
```

`samurai::petsc::solve` uses a linear solver for the linear scheme types and a non-linear (Newton) solver for `SchemeType::NonLinear`.
Both solvers require boundary conditions attached to the unknown field; the {doc}`PETSc how-to guide <../howto/petsc>` describes the solvers.
The discrete operator is named "(unnamed)" unless you call `D.set_name("...")`; the name appears in error messages and timers.

```{note}
The cells and field values passed to the flux function are those of the *computational* stencil, not the two real cells around the face.
Where a level jump occurs, at least one of the computational cells is a ghost.
The ghosts of the input field must therefore hold values.
`D(u)` updates them before it computes the fluxes, if they are out of date:
it calls `samurai::update_ghost` on an AMR mesh and `samurai::update_ghost_mr` on the other meshes.
The implicit assembly of a non-linear or heterogeneous scheme does the same for the unknown field.
The ghosts of any other field the flux function reads are updated only if this field is declared as a [parameter field](#parameter-fields).
```

(fv-available-implementations)=

## Available implementations

The following functions are declared in `samurai/schemes/fv.hpp` and the headers it includes.
`Field` stands for the type of the field the discrete operator applies to.

| Function | Discrete operator | Scheme type | Stencil size |
| --- | --- | --- | --- |
| `make_flux_based_scheme(flux_definition)` | user-defined flux | from `cfg` | from `cfg` |
| `make_divergence(flux_definition)` | alias of `make_flux_based_scheme` | from `cfg` | from `cfg` |
| `make_cell_based_scheme<cfg>()` | user-defined local scheme (see {doc}`local_schemes`) | from `cfg` | from `cfg` |
| `make_diffusion_order2<Field>(...)` | $-\nabla\cdot(K\nabla u)$ | linear homogeneous or heterogeneous | 2 |
| `make_multi_diffusion_order2<Field>(K)` | $-\nabla\cdot(K_i\nabla u_i)$ per component | linear homogeneous | 2 |
| `make_laplacian_order2<Field>()` | $\Delta u$ | linear homogeneous | 2 |
| `make_diffusion_cell_based<Field>()` | $-\Delta u$, cell-based | linear homogeneous | star, radius 1 |
| `make_diffusion_cell_based_nonlin<Field>()` | $-\Delta u$, cell-based | non-linear | star, radius 1 |
| `make_convection_upwind<Field>(...)` | convection, upwind | linear or non-linear | 2 |
| `make_convection_weno5<Field>(...)` | convection, WENO5 | non-linear | 6 |
| `make_convection_smooth_rusanov_incompressible<Field>(...)` | convection, smooth Rusanov | non-linear | 2 |
| `make_dual_convection_smooth_rusanov_incompressible<Field, VelocityField>(u)` | Jacobian of the smooth Rusanov convection w.r.t. the velocity | non-linear | 2 |
| `make_gradient_order2<Field>()` | $\nabla u$ | linear homogeneous | 2 |
| `make_divergence_order2<Field>()` | $\nabla\cdot\mathbf{u}$ | linear homogeneous | 2 |
| `make_identity<Field>()` | identity | linear homogeneous | 1 |
| `make_zero_operator<Field>()`, `make_zero_operator<OutputField, InputField>()` | zero | linear homogeneous | 1 |
| `make_buoyancy<VelocityField, TemperatureField>(factor)` | $T \mapsto -\text{factor}\, T\, \mathbf{e}_y$ | linear homogeneous | 1 |
| `make_operator_sum(operators...)` | sum of discrete operators that cannot be merged | from the summed discrete operators | largest among the summed discrete operators |
| `make_gradient<Field>()` | **deprecated**: use `make_gradient_order2` | | |
| `make_divergence<Field>()` (no argument) | **deprecated**: use `make_divergence_order2` | | |

All functions are in the `samurai` namespace.
The next sections give details and examples.

### Diffusion operators

`samurai::make_diffusion_order2` implements the diffusion operator $-\nabla\cdot(K\nabla u)$ with the classical second-order scheme.
It applies to scalar and vector fields.
The overload depends on the type of coefficient.

Homogeneous, scalar coefficient:

```c++
double K  = 10;
auto diff = samurai::make_diffusion_order2<FieldType>(K);
```

Without argument, the coefficient is 1.

Homogeneous, diagonal tensor coefficient:

```c++
static constexpr std::size_t dim = 2;

samurai::DiffCoeff<dim> K;
K(0) = 1; // x-direction
K(1) = 2; // y-direction

auto diff = samurai::make_diffusion_order2<FieldType>(K);
```

Heterogeneous, diagonal tensor coefficient, stored in a scalar field whose value type is `samurai::DiffCoeff<dim>`:

```c++
static constexpr std::size_t dim = 2;

using coeff_t   = samurai::DiffCoeff<dim>;
using mesh_id_t = decltype(mesh)::mesh_id_t;

auto K = samurai::make_scalar_field<coeff_t>("K", mesh);

auto set_K = [&](auto& cell)
{
    double x = cell.center(0);
    K[cell]  = x < 0 ? coeff_t{1, 2} : coeff_t{2, 1};
};
samurai::for_each_cell(mesh[mesh_id_t::reference], set_K);

auto diff = samurai::make_diffusion_order2<FieldType>(K);
```

This diffusion operator is `SchemeType::LinearHeterogeneous` and does not declare `K` as a parameter field:
the loop over the `reference` cells sets `K` in the ghosts too.
On each face, the scheme uses the value of `K` in the left (or bottom, or back) cell of the stencil.

One coefficient per field component, for a vector field:

```c++
samurai::DiffCoeff<2> K;
K(0)      = 1;    // first component
K(1)      = 0.01; // second component
auto diff = samurai::make_multi_diffusion_order2<VectorFieldType>(K);
```

Laplacian operator, equal to the diffusion operator with the opposite sign and the coefficient 1:

```c++
auto lap = samurai::make_laplacian_order2<FieldType>();
```

`samurai::make_diffusion_cell_based<FieldType>()` and `samurai::make_diffusion_cell_based_nonlin<FieldType>()` implement $-\Delta u$ as a cell-based scheme on a star stencil of radius 1,
with the coefficient $2\,\text{dim}/h^2$ on the center cell and $-1/h^2$ on its neighbours.
The first one is linear, the second one is non-linear and provides its Jacobian function.

### Convection operators

The convection operators are named `samurai::make_convection_<scheme><FieldType>(...)`, where `<scheme>` is the discrete scheme:

- `upwind`: first-order upwind scheme;
- `weno5`: fifth-order WENO scheme of Jiang and Shu [(1996)](#references), with a stencil of size 6;
- `smooth_rusanov_incompressible`: Rusanov scheme whose numerical viscosity uses a smooth approximation of the absolute value,
  $\lambda = \sqrt{v^2 + \varepsilon^2}$ with $\varepsilon = 10^{-6}$, so that the flux is differentiable.

The discrete operator implemented is $\nabla \cdot (\mathbf{a} \otimes u)$, which equals $\mathbf{a}\cdot\nabla u$ if $\mathbf{a}$ is divergence-free.

Linear convection with a constant velocity (`upwind` and `weno5`):

```c++
samurai::VelocityVector<dim> a = {1, -2};

auto conv = samurai::make_convection_upwind<FieldType>(a);
```

Linear convection with a velocity field (`upwind`, `weno5` and `smooth_rusanov_incompressible`):

```c++
// One component per space dimension
auto a    = samurai::make_vector_field<double, dim>("a", mesh);
auto conv = samurai::make_convection_upwind<FieldType>(a);
```

The `upwind` and `weno5` overloads register the velocity field as a {ref}`parameter field <parameter_field>`: the discrete operator updates its ghosts.
The `smooth_rusanov_incompressible` overload does not: its velocity ghosts must be updated by the caller.

Non-linear convection $\nabla \cdot (\mathbf{u} \otimes \mathbf{u})$ (`upwind`, `weno5` and `smooth_rusanov_incompressible`):

```c++
auto conv = samurai::make_convection_upwind<VelocityField>();
```

For `upwind` and `weno5`, the field must be a scalar field or have as many components as the space dimension;
`smooth_rusanov_incompressible` requires as many components as the space dimension.

Not every convection operator provides a Jacobian function, which an implicit solve needs (see [Jacobian function](#jacobian-function)):

| Function | Arguments | Scheme type | Jacobian function |
| --- | --- | --- | --- |
| `make_convection_upwind` | constant velocity | linear homogeneous | not needed |
| `make_convection_upwind` | velocity field | linear heterogeneous | not needed |
| `make_convection_upwind` | none | non-linear | yes (vector fields: only for `dim <= 2`) |
| `make_convection_weno5` | constant velocity | non-linear | yes |
| `make_convection_weno5` | velocity field | non-linear | no |
| `make_convection_weno5` | none | non-linear | no |
| `make_convection_smooth_rusanov_incompressible` | velocity field | non-linear | yes |
| `make_convection_smooth_rusanov_incompressible` | none | non-linear | yes |

`samurai::make_dual_convection_smooth_rusanov_incompressible<FieldType, VelocityField>(u)` defines only a Jacobian function:
the derivative of the smooth Rusanov convection of the scalar field `u` with respect to the velocity.
It fills an off-diagonal block of the Jacobian matrix of a non-linear block operator,
as in `demos/FiniteVolume/diff_heated_cavity.cpp`.

### Gradient operator

The gradient operator is implemented for scalar fields only (fields with one component), with the centered scheme of order 2.

```c++
auto grad = samurai::make_gradient_order2<FieldType>();
```

### Divergence operator

The divergence operator is implemented only for vector fields with as many components as the space dimension, with the centered scheme of order 2.

```c++
auto div = samurai::make_divergence_order2<VelocityField>();
```

### Identity operator

The identity operator is used in implicit time-stepping schemes.
For the heat equation, with a backward Euler scheme:

```c++
auto u    = samurai::make_scalar_field<double>("u", mesh);
auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

auto diff = samurai::make_diffusion_order2<decltype(u)>();
auto id   = samurai::make_identity<decltype(u)>();

// Solves the linear equation [id + dt * diff](unp1) = u
samurai::petsc::solve(id + dt * diff, unp1, u);
```

Discrete operators combine with `+`, `-` and the multiplication by a scalar.
Two flux-based schemes with the same configuration merge into one;
otherwise the result is a `samurai::OperatorSum`, as built by `samurai::make_operator_sum`.

### Zero operator

The zero operator fills the zero blocks of block operators.
`samurai::make_zero_operator<Field>()` maps `Field` to `Field`; `samurai::make_zero_operator<OutputField, InputField>()` maps `InputField` to `OutputField`.
For the Stokes system:

```c++
// Unknowns
auto velocity = samurai::make_vector_field<double, dim>("u", mesh);
auto pressure = samurai::make_scalar_field<double>("p", mesh);

// Stokes operator
auto diff = samurai::make_diffusion_order2<decltype(velocity)>();
auto grad = samurai::make_gradient_order2<decltype(pressure)>();
auto div  = samurai::make_divergence_order2<decltype(velocity)>();
auto zero = samurai::make_zero_operator<decltype(pressure)>();

auto stokes = samurai::make_block_operator<2, 2>(diff, grad,
                                                  -div, zero);
// Right-hand side
auto f = samurai::make_vector_field<double, dim>("f", mesh);
auto z = samurai::make_scalar_field<double>("z", mesh, 0.);

// Linear solver
auto stokes_solver = samurai::petsc::make_solver(stokes);
stokes_solver.set_unknowns(velocity, pressure);
stokes_solver.solve(f, z);
```

### Buoyancy operator

`samurai::make_buoyancy<VelocityField, TemperatureField>(factor)` maps a scalar temperature field $T$ to the velocity field $-\text{factor}\, T\, \mathbf{e}_y$, where $\mathbf{e}_y$ is the unit vector of the y-direction.
`factor` defaults to 1.
The discrete operator is not implemented in 1D.

```c++
auto buoyancy =
    samurai::make_buoyancy<VelocityField, TemperatureField>(9.81);
```

## Errors

These errors are raised when a discrete operator is applied or assembled.

The stencil size of the scheme exceeds the max stencil size of the mesh, when the discrete operator is applied explicitly (`std::invalid_argument`):

```text
The stencil size required by the scheme 'convection' (6) is larger than the max_stencil_size parameter of the mesh (4).
You can set it with mesh_config.max_stencil_radius(3) or mesh_config.max_stencil_size(6).
```

The max stencil size of a mesh is twice its max stencil radius, which is at least 2 unless the mesh configuration calls `disable_minimal_ghost_width()`.
Set `max_stencil_radius` in the mesh configuration (or `--max-stencil-radius` on the command line) to the stencil radius of the widest scheme, 3 for WENO5.

A non-linear scheme without Jacobian function is solved implicitly (`std::runtime_error`, raised when the Jacobian matrix is assembled):

```text
The jacobian function of operator 'burgers' has not been implemented.
Use option -snes_mf or -snes_fd for an automatic computation of the jacobian matrix.
```

Define `cons_jacobian_function` (or `jacobian_function` for a non-conservative scheme), or pass `-snes_mf` to the program (the message also names `-snes_fd`, which fails with the default direct solver, see [#625](https://github.com/hpc-maths/samurai/issues/625)).

Mismatched field types in the ready-made discrete operators fail at compile time with a `static_assert`, for instance
"The field type for the divergence operator must have a size equal to the space dimension."

`make_flux_based_scheme` also prints a warning when the stencil size is larger than 4 in 2D or 3D, the option `--finer-level-flux` is not 0 (fluxes computed at finer levels) and the option `--refine-boundary` is not set:

```text
Warning: for stencils larger than 4, computing fluxes at max_level may cause issues close to the boundary.
```

## References

G.-S. Jiang and C.-W. Shu.
Efficient implementation of weighted ENO schemes.
*Journal of Computational Physics*, 126(1):202-228, 1996.
DOI: [10.1006/jcph.1996.0130](https://doi.org/10.1006/jcph.1996.0130).
