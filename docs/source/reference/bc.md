# Boundary conditions

A boundary condition attached to a field gives the values of the field's ghosts outside the domain.
{{ project }} provides Dirichlet and Neumann conditions, applies a polynomial extrapolation to the ghosts that no condition reaches, and accepts user-defined conditions.
A periodic direction has no boundary condition: periodicity is a property of the mesh.
{doc}`../howto/bc` shows how to set each kind of condition, with examples that print the ghost values.

The conditions are declared in `<samurai/bc.hpp>`.
The lattice Boltzmann conditions are declared in `<samurai/schemes/lbm/boundary.hpp>`.

## Attachment to a field

{cpp:func}`samurai::make_bc` builds a condition and attaches it to a field.
The condition type is a template argument; the value follows the field.

| Call | Value imposed |
| --- | --- |
| `samurai::make_bc<BC>(u)` | 0 for every component |
| `samurai::make_bc<BC>(u, v1, ..., vn)` | the constants `v1` to `vn`, one per component |
| `samurai::make_bc<BC>(u, f)` | the value returned by the function `f` |

`BC` is either a tag such as `samurai::Dirichlet<1>`, which exposes the condition class through a member alias template `impl_t<Field>`, or a class template with a single `Field` parameter (see {ref}`reference-bc-custom`).

```c++
auto u = samurai::make_scalar_field<double>("u", mesh);
samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

auto v = samurai::make_vector_field<double, 3>("v", mesh);
samurai::make_bc<samurai::Dirichlet<1>>(v, 1., 2., 3.);
```

### Constant values

The number of constants must equal the number of components of the field, and their common type must be the value type of the field.
Both rules are checked at compile time.

### Value function

The function has the signature

```c++
value_t f(const samurai::DirectionVector<dim>& direction, const cell_t& cell_in, const coords_t& coords);
```

- `direction` is the outward unit vector of the boundary, for example `{-1, 0}` for the left boundary in 2D.
- `cell_in` is the {cpp:class}`samurai::Cell` inside the domain that touches the boundary.
- `coords` is an `xt::xtensor_fixed<double, xt::xshape<dim>>` holding the center of the boundary face of `cell_in` in `direction`.
- The return value is the value type of the field for a scalar field, and a `samurai::Array<value_type, n_comp>` for a vector field.

The function is evaluated each time the ghosts are filled, so it may capture variables such as the current time.

```c++
samurai::make_bc<samurai::Dirichlet<1>>(u,
                                        [](const auto&, const auto&, const auto& x)
                                        {
                                            return x[0] * x[0];
                                        });
```

### Return value and errors

{cpp:func}`samurai::make_bc` returns a pointer to the condition stored in the field, of type `samurai::Bc<Field>*`.
The field owns the condition; the pointer is used to restrict it to a region with `->on(...)`.

A condition fills `stencil_size / 2` ghost layers.
If its stencil size exceeds the `max_stencil_size` of the mesh configuration, {cpp:func}`samurai::make_bc` throws `std::invalid_argument`:

```text
The stencil size required by this boundary condition (6) is larger than the max_stencil_size parameter of the mesh (4).
You can set it with mesh_config.max_stencil_radius(3) or mesh_config.max_stencil_size(6).
```

### Several conditions on one field

A field holds a list of conditions.
Each call to {cpp:func}`samurai::make_bc` appends one.
They are applied in the order they were attached: where two regions overlap, the condition attached last sets the ghost values.

Copying a field or assigning a field to another copies its conditions.
`u.copy_bc_from(other)` appends copies of the conditions of `other` to those of `u`.
`u.get_bc()` returns the list.

## Built-in conditions

| Condition | Orders | Stencil size | Ghost layers filled |
| --- | --- | --- | --- |
| `samurai::Dirichlet<order>` | 1 to 4 | `2 * order` | `order` |
| `samurai::Neumann<order>` | 1 | 2 | 1 |

`order` defaults to 1.
An order outside these ranges fails at compile time with a `static_assert`.

With the default mesh configuration, `max_stencil_size` is 4: `samurai::Dirichlet<1>`, `samurai::Dirichlet<2>` and `samurai::Neumann<1>` fit within it.
`samurai::Dirichlet<3>` and `samurai::Dirichlet<4>` need `mesh_config.max_stencil_radius(order)`.

### samurai::Dirichlet

`samurai::Dirichlet<order>` imposes the value {math}`v` of the field on the boundary face.
It builds the polynomial {math}`p` of degree `order` that takes the values of the `order` inner cells next to the boundary at their centers and the value {math}`v` at the boundary face.
Each ghost receives {math}`p` at its center.

For `order = 1`, with {math}`u_0` the boundary cell and {math}`u_g` the ghost,

$$
u_g = 2v - u_0.
$$

```{remark}
The construction treats the cell values as point values at the cell centers.
A finite volume field stores cell averages: on cell averages, `samurai::Dirichlet<order>` reproduces polynomials of degree 1 exactly, whatever the order.
`tests/test_bc_ghost_values.cpp` checks both conventions.
```

### samurai::Neumann

`samurai::Neumann<1>` imposes the derivative {math}`v` of the field along the **outward** normal {math}`\mathbf{n}`:

$$
v = \frac{\partial u}{\partial \mathbf{n}}, \qquad u_g = u_0 + \Delta x \, v,
$$

where {math}`\Delta x` is the length of the cell.
On the left boundary of a 1D domain, {math}`\mathbf{n} = -\mathbf{e}_x` and {math}`v = -\partial u / \partial x`.
For {math}`u(x) = x` on {math}`[0, 1]`, the value that matches the solution is {math}`v = -1` on the left boundary and {math}`v = 1` on the right boundary.
`samurai::make_bc<samurai::Neumann<1>>(u)` imposes a zero normal derivative.

### Lattice Boltzmann conditions

`samurai::BounceBack`, `samurai::AntiBounceBack` and `samurai::ImposedDistribution` are conditions for the distribution functions of a lattice Boltzmann scheme.
Their `make_bc` overloads take the lattice velocities or the imposed distribution; their parameters are documented in `include/samurai/schemes/lbm/boundary.hpp`.

## Regions

By default a condition applies to the whole boundary.
`->on(...)` on the pointer returned by {cpp:func}`samurai::make_bc` restricts it to a region, and returns the same pointer.

| Argument of `on` | Region |
| --- | --- |
| one or more `samurai::DirectionVector<dim>` | the boundary faces whose outward normal is one of the given directions |
| a `std::array` of `samurai::DirectionVector<dim>` | same, from an array |
| a set expression | the cells of the set that touch the boundary, in every Cartesian direction where they touch it |
| `samurai::make_bc_region(mesh, predicate)` | the boundary faces, in the Cartesian directions, whose center `x` satisfies `predicate(x)` |

The predicate takes an `xt::xtensor_fixed<double, xt::xshape<dim>>` and returns `bool`.
A diagonal direction such as `{-1, -1}` (the bottom-left corner) is accepted; it has an effect only on a condition that fills diagonal directions (see [Ghosts beyond the condition](#ghosts-beyond-the-condition)).

The demo `demos/FiniteVolume/diff_heated_cavity.cpp` uses directions:

```{literalinclude} ../../../demos/FiniteVolume/diff_heated_cavity.cpp
  :language: c++
  :start-at: samurai::DirectionVector<dim> left
  :end-at: ->on(top, bottom);
  :dedent:
```

A predicate selects the upper half of the boundary:

```c++
samurai::make_bc<samurai::Dirichlet<1>>(u, 1.)->on(samurai::make_bc_region(mesh,
                                                                             [](const auto& x)
                                                                             {
                                                                                 return x[1] > 0.5;
                                                                             }));
```

A set expression selects the cells of the left boundary:

```c++
const samurai::DirectionVector<dim> left = {-1, 0};
auto left_cells = samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), -left));
samurai::make_bc<samurai::Neumann<1>>(u, 0.)->on(left_cells);
```

The corner cells of `left_cells` also touch the bottom and top boundaries, so the condition applies there as well.

## Periodicity

A periodic direction is set on the mesh configuration, not with a boundary condition:

```c++
auto config   = samurai::mesh_config<dim>().periodic(true);          // every direction
auto config_x = samurai::mesh_config<dim>().periodic({true, false}); // x only
```

The ghosts of a periodic direction take the values of the cells on the opposite side of the domain.
A condition attached to the field is not applied in a periodic direction, nor in a diagonal direction that has a periodic component.

## Ghosts beyond the condition

The ghost width of a mesh can exceed the number of layers a condition fills: it is set by the max stencil radius of the mesh, which is raised to 2 unless the mesh configuration calls `disable_minimal_ghost_width()`.
{{ project }} fills the remaining ghosts by polynomial extrapolation, implemented by `samurai::PolynomialExtrapolation<Field, stencil_size>`:

- The layers beyond the largest `stencil_size / 2` of the attached conditions are filled one at a time, from the closest to the farthest.
  The stencil ends on the ghost to fill and grows with the layer up to 6 cells (a polynomial of degree 4); beyond, it slides outward at constant size and uses the ghosts already filled.
  The extrapolation reaches 5 ghost layers.
  A mesh with a larger ghost width throws `std::runtime_error` when its ghosts are updated.
- The ghosts in the diagonal directions (the corners in 2D, the edges and corners in 3D) receive the mirror image of the inner diagonal cells about the corner, and the off-diagonal ghosts of the corner block copy the diagonal value.
  A condition replaces this treatment only when it fills diagonal directions itself (`fills_diagonal_directions()` returns `true`), as the lattice Boltzmann reflections do for velocity sets with diagonal velocities.

`samurai::PolynomialExtrapolation` is not attached with {cpp:func}`samurai::make_bc`: it has two template parameters, and {{ project }} applies it on its own.

## Application of the conditions

The conditions fill the ghosts during a ghost update: {cpp:func}`samurai::update_ghost_mr`, {cpp:func}`samurai::update_ghost` and {cpp:func}`samurai::update_outer_ghosts`.
Explicit finite volume schemes call `samurai::update_ghost_if_needed` on their input field before they evaluate it.

On an adapted mesh, a ghost update applies the conditions at every level that holds cells.
The outer ghosts of a level are also filled from the finer level, as the average of their children, and from the coarser level by prediction of order 0.

The implicit schemes assembled with PETSc read the conditions to write the boundary equations into the matrix.
They assemble `samurai::Dirichlet` and `samurai::Neumann` only, with the order equal to the number of boundary ghosts of the scheme.
Any other condition is skipped with the message `Unknown boundary condition type. Only Dirichlet and Neumann are implemented at the moment.`

(reference-bc-custom)=

## Custom conditions

A condition is a class derived from {cpp:class}`samurai::Bc` that defines:

- `get_stencil`: the cells the condition reads and writes, for the right boundary (direction {math}`+\mathbf{e}_x`). {{ project }} rotates it for the other directions. The origin `{0, ..., 0}` is the boundary cell; the default is `samurai::line_stencil<dim, 0, stencil_size>()`, which runs from `-stencil_size / 2 + 1` to `stencil_size / 2`.
- `get_apply_function`: a function that receives the field, the `std::array` of the cells captured by the stencil and the value of the condition, and writes the ghosts.

The macro `INIT_BC(Name, stencil_size)` declares the constructors and the types `stencil_t`, `constant_stencil_size_t`, `stencil_cells_t`, `apply_function_t`, `value_t` and `direction_t`.
The stencil size is between 1 and 10.

The built-in conditions pair an implementation class template with a tag.
The example below follows the same design: the ghost receives the value of the boundary cell plus the value of the condition.

```c++
template <class Field>
struct ExtrapolationImpl : public samurai::Bc<Field>
{
    INIT_BC(ExtrapolationImpl, 2)

    stencil_t get_stencil(constant_stencil_size_t) const override
    {
        return samurai::line_stencil<dim, 0>(0, 1);
    }

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t& value)
        {
            u[cells[1]] = u[cells[0]] + value;
        };
    }
};

struct Extrapolation
{
    template <class Field>
    using impl_t = ExtrapolationImpl<Field>;
};

samurai::make_bc<Extrapolation>(u, 0.);
```

`samurai::line_stencil<dim, 0>(0, 1)` captures the boundary cell (`cells[0]`) and the ghost on its right (`cells[1]`).
The second argument of `get_apply_function` is the outward direction of the boundary being filled.
The {ref}`stencil configuration <stencil-configuration>` section of the finite volume reference describes stencils in more detail.

{cpp:func}`samurai::make_bc` also accepts the implementation class template directly, as in `samurai::make_bc<Mybc>(u, 0.)` in the demo {download}`advection_2d_user_bc.cpp <../../../demos/FiniteVolume/advection_2d_user_bc.cpp>`.
