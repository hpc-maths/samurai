# Finite volume schemes

This page documents the finite volume schemes of {{ project }}: the ready-made discrete operators built by the `samurai::make_*` functions, the configuration types that describe a user-defined scheme, and the functions that combine schemes.
A flux-based scheme is described by a `samurai::FluxConfig` and a `samurai::FluxDefinition`, then built by `samurai::make_flux_based_scheme`; a cell-based scheme is described by a `samurai::CellBasedSchemeConfig` and built by `samurai::make_cell_based_scheme`.
The resulting discrete operator applies to a field in an explicit context (`auto v = D(u);`) or goes to a PETSc solver in an implicit context (see {doc}`petsc`).
Include `samurai/schemes/fv.hpp` to get all of them.
For the definitions and examples, see the {doc}`finite volume schemes reference <../reference/finite_volume_schemes>` and the {doc}`local schemes reference <../reference/local_schemes>`.

## Scheme type

`samurai::SchemeType` tells how the scheme depends on the field it applies to, and therefore which function defines it and which solver `samurai::petsc::solve` picks.

```{doxygenenum} samurai::SchemeType
```

## Flux-based schemes

`samurai::FluxConfig` declares the scheme type, the stencil size, the output and input field types, and an optional parameter field type.

```{doxygenstruct} samurai::FluxConfig
:members:
:undoc-members:
```

`samurai::FluxDefinition<cfg>` holds one flux definition per Cartesian direction: `flux[d]` is the definition in the direction `d`.
Each one stores the direction and the stencil of the flux, from `samurai::NormalFluxDefinitionBase`, and the flux function, whose member and signature depend on the scheme type:

| Scheme type | Member to set | Signature |
| --- | --- | --- |
| `LinearHomogeneous` | `cons_flux_function` | `void(FluxStencilCoeffs<cfg>&, double h)` |
| `LinearHeterogeneous` | `cons_flux_function` | `void(FluxStencilCoeffs<cfg>&, const StencilData<cfg>&)` |
| `NonLinear` | `cons_flux_function` or `flux_function` | `void(FluxValue<cfg>&, const StencilData<cfg>&, const StencilValues<cfg>&)`, or `void(FluxValuePair<cfg>&, ...)` for a non-conservative scheme |

A non-linear scheme solved implicitly also sets `cons_jacobian_function` or `jacobian_function`.
These members belong to the specializations of `samurai::NormalFluxDefinition`, which Doxygen merges into one entry; this reference does not render them.

```{doxygenclass} samurai::FluxDefinition
:members:
:undoc-members:
```

```{doxygenclass} samurai::NormalFluxDefinitionBase
:members:
```

```{doxygenstruct} samurai::StencilData
:members:
:undoc-members:
```

```{doxygenfunction} samurai::make_flux_based_scheme
```

```{doxygenfunction} samurai::make_divergence(const FluxDefinition<cfg> &flux_definition)
```

## Cell-based schemes

`samurai::CellBasedSchemeConfig` declares a scheme that computes the value of a cell from a stencil of cells around it.
`samurai::StarStencilSchemeConfig` sets the stencil to the center cell and `neighbourhood_width` cells in each Cartesian direction; `samurai::LocalCellSchemeConfig` is the star stencil of width 0, which reads only the cell itself.

```{doxygenstruct} samurai::CellBasedSchemeConfig
:members:
:undoc-members:
```

```{doxygentypedef} samurai::StarStencilSchemeConfig
```

```{doxygentypedef} samurai::LocalCellSchemeConfig
```

`make_cell_based_scheme<cfg>()` uses the boundary configuration `samurai::BoundaryConfigFV<1>`; pass a second template argument to choose another one.
The overloads that take a name set the name of the discrete operator.

```{doxygenfile} cell_based_scheme.hpp
:sections: innernamespace func
```

```{doxygenstruct} samurai::BoundaryConfigFV
:members:
:undoc-members:
```

## Diffusion operators

`samurai::DiffCoeff<n>` holds a diagonal diffusion tensor, one coefficient per direction, or one per component for `make_multi_diffusion_order2`.

```{doxygentypedef} samurai::DiffCoeff
```

```{doxygenfunction} samurai::make_diffusion_order2(const DiffCoeff<Field::dim> &K)
```

```{doxygenfunction} samurai::make_diffusion_order2(double k)
```

```{doxygenfunction} samurai::make_diffusion_order2()
```

```{doxygenfunction} samurai::make_diffusion_order2(const DiffTensorField &K)
```

```{doxygenfunction} samurai::make_multi_diffusion_order2
```

```{doxygenfunction} samurai::make_laplacian_order2
```

```{doxygenfunction} samurai::make_diffusion_cell_based
```

```{doxygenfunction} samurai::make_diffusion_cell_based_nonlin
```

## Convection operators

`samurai::VelocityVector<dim>` holds a constant velocity.
`make_convection_upwind` and `make_convection_weno5` have an overload for a constant velocity, one for a velocity field, and one without argument for the non-linear convection of the field by itself; `make_convection_smooth_rusanov_incompressible` has the last two.

```{doxygentypedef} samurai::VelocityVector
```

```{doxygenfunction} samurai::make_convection_upwind(const VelocityVector<Field::dim> &velocity)
```

```{doxygenfunction} samurai::make_convection_upwind(VelocityField &velocity_field)
```

```{doxygenfunction} samurai::make_convection_upwind()
```

```{doxygenfunction} samurai::make_convection_weno5(const VelocityVector<Field::dim> &velocity)
```

```{doxygenfunction} samurai::make_convection_weno5(VelocityField &velocity_field)
```

```{doxygenfunction} samurai::make_convection_weno5()
```

```{doxygenfunction} samurai::make_convection_smooth_rusanov_incompressible(VelocityField &velocity_field)
```

```{doxygenfunction} samurai::make_convection_smooth_rusanov_incompressible()
```

```{doxygenfunction} samurai::make_dual_convection_smooth_rusanov_incompressible
```

## Other operators

`make_gradient()` and `make_divergence()` without argument are deprecated: use `make_gradient_order2()` and `make_divergence_order2()`.
`make_zero_operator<Field>()` maps `Field` to `Field`; `make_zero_operator<OutputField, InputField>()` maps `InputField` to `OutputField`.

```{doxygenfunction} samurai::make_gradient_order2
```

```{doxygenfunction} samurai::make_divergence_order2
```

```{doxygenfunction} samurai::make_identity
```

```{doxygenfile} zero_operator.hpp
:sections: innernamespace func
```

```{doxygenfunction} samurai::make_buoyancy
```

```{doxygenfunction} samurai::make_gradient
```

```{doxygenfunction} samurai::make_divergence()
```

## Combine schemes

Discrete operators combine with `+`, `-` and the multiplication by a scalar.
Two flux-based schemes with the same configuration merge into one; otherwise the result is a `samurai::OperatorSum`, which `make_operator_sum` also builds.
`make_block_operator<rows, cols>(...)` arranges discrete operators, given row by row, into a block operator for a system of equations.
`make_explicit` returns the `samurai::Explicit` object that `D(u)` uses to apply a scheme explicitly.

```{doxygenfunction} samurai::make_operator_sum(const Operators&... operators)
```

```{doxygenfunction} samurai::make_operator_sum(const std::tuple<Operators...> &operator_tuple)
```

```{doxygenfunction} samurai::make_block_operator
```

```{doxygenfunction} samurai::make_explicit
```
