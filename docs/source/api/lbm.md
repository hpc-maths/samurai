# Lattice Boltzmann schemes

This page documents the lattice Boltzmann schemes of {{ project }}.
A `samurai::LBMScheme` is a list of elementary velocity schemes (`samurai::VelocityScheme`), each one a group of lattice velocities with its moment matrices, relaxation rates and equilibrium.
The scheme works on two vector fields on the same mesh: the distributions `f`, which are the unknowns, and the moments `m`, on which the mesh is adapted and the results are saved.
The wall boundary conditions `samurai::BounceBack`, `samurai::AntiBounceBack` and `samurai::ImposedDistribution` attach to `f` with `samurai::make_bc`, like the finite volume boundary conditions.
Include `samurai/schemes/lbm.hpp` to get all of them.
For a scheme written step by step, see the {doc}`LBM schemes guide <../LBM/schemes>`; for the programs of `demos/LBM`, see the {doc}`LBM test cases <../LBM/test_cases>`.

## Velocity schemes

`samurai::VelocityScheme<dim, q>` holds `q` velocities in dimension `dim`:

| Member | Content |
| --- | --- |
| `velocities` | the lattice velocities, as integer vectors |
| `M` | the matrix that maps the distributions to the moments, row-major |
| `invM` | the inverse of `M` |
| `s` | the relaxation rate of each moment, 0 for a conserved moment |
| `equilibrium` | the function `void(moments_t& meq, std::span<const double> m_all)` that fills the `q` equilibrium moments of this block |

`equilibrium` receives the moments of every block of the scheme, concatenated, so that the equilibrium of one block can depend on the conserved moments of the others.
`velocity_scheme<dim, q>(velocities, M, invM, s, equilibrium)` builds a `VelocityScheme`.

```{doxygenstruct} samurai::VelocityScheme
:members:
:undoc-members:
```

```{doxygenfunction} samurai::velocity_scheme
```

## Scheme

`make_lbm_scheme<Field>(name, lambda, blocks...)` builds an `LBMScheme` from one or more velocity schemes.
`Field` is the type of `f` and `m`: the dimension of every block must be the dimension of the field, and the sum of the `q` of the blocks must be the number of components of the field.
The scheme stores the lattice velocity `lambda` and returns it with `lambda()`; the time step does not read it.

`init_equilibrium(f, m)` sets `f` to the equilibrium distribution of the moments in `m`: it keeps the conserved moments of `m` and replaces the others by their equilibrium value.
A call `scheme(f, m, dt)` advances one time step:

1. it updates the ghosts of `f` with `samurai::update_ghost_mr`, which applies the boundary conditions;
2. it streams `f` on every level of the adapted mesh;
3. it computes the moments, relaxes each one towards its equilibrium with `m_k += s_k (meq_k - m_k)`, applies the source term if one is set, and computes `f` back from the moments;
4. it stores the moments in `m`.

`set_source(source)` registers a source term, such as gravity, called on the moments of each cell with `dt` after the relaxation.
`dt` is used only by the source term and defaults to 0.

```{doxygenclass} samurai::LBMScheme
:members:
:undoc-members:
```

```{doxygenfunction} samurai::make_lbm_scheme
```

## Boundary conditions

The three boundary conditions fill the outer ghost cell next to the boundary, which the stream then reads.
Each one is a tag passed as template argument to `make_bc`:

| Tag | Ghost value | Effect |
| --- | --- | --- |
| `BounceBack` | `f_inner(opposite[a]) + rhs(a)` | Imposes the odd moments, such as the momentum: a no-slip wall, or a moving wall with `rhs`. |
| `AntiBounceBack` | `-f_inner(opposite[a]) + rhs(a)` | Imposes the even moments, such as the density, the pressure or the water height. |
| `ImposedDistribution` | a fixed distribution | Pulls this distribution into the domain: an inflow. |

`opposite[a]` is the index of the velocity opposite to velocity `a`, or `a` itself for the rest velocity.
`rhs` is zero when no wall equilibrium is given.
Given an equilibrium distribution `feq` to reflect around, `rhs(a) = feq(a) - feq(opposite[a])` for `BounceBack` and `rhs(a) = feq(a) + feq(opposite[a])` for `AntiBounceBack`.

The `make_bc` overloads for `BounceBack` and `AntiBounceBack` take:

- the lattice velocities, the same list as in `velocity_scheme`;
- optionally the wall equilibrium, either a constant distribution such as `scheme.equilibrium_f(...)`, or a callable that takes the distribution of the inner cell and returns the equilibrium to reflect around;
- for a scheme with several blocks, the sizes of the blocks and, for each block, the axis of the momentum component it carries, or -1 for a scalar such as the density or the energy.
  The opposite velocity is then searched within each block, and the block that carries the momentum normal to the wall changes sign: a slip wall.

`ImposedDistribution` uses the overload with one argument after the field, the distribution to impose.

The shallow water program `demos/LBM/new_D1Q3_shallow_waters_dam.cpp` closes the left side with a wall and imposes the water height `hR` on the right side, with a callable that keeps the momentum of the inner cell:

```{literalinclude} ../../../demos/LBM/new_D1Q3_shallow_waters_dam.cpp
:language: c++
:start-at: auto reservoir = [&scheme](double h_wall)
:end-at: make_bc<samurai::AntiBounceBack>
:dedent: 8
```

The Euler program `demos/LBM/new_D2Q4444_euler_implosion.cpp` has four blocks of four velocities, for the density, the two momentum components and the energy, and puts a slip wall on every side:

```{literalinclude} ../../../demos/LBM/new_D2Q4444_euler_implosion.cpp
:language: c++
:start-at: std::array<std::array<int, dim>, 16> velocities{};
:end-at: make_bc<samurai::BounceBack>
:dedent: 4
```

These overloads apply only to tags that declare `lbm_bc_tag`, so they never compete with the `make_bc` of the finite volume boundary conditions, which also apply to `f` (see {doc}`bc`).

```{doxygenfunction} samurai::make_bc(Field &field, const Vel &velocities)
```

```{doxygenfunction} samurai::make_bc(Field &field, const Vel &velocities, const Feq &f_wall)
```

```{doxygenfunction} samurai::make_bc(Field &field, const Vel &velocities, const std::vector<std::size_t> &block_sizes, const std::vector<int> &block_odd_axis)
```

```{doxygenfunction} samurai::make_bc(Field &field, const Vel &velocities, const Feq &f_wall, const std::vector<std::size_t> &block_sizes, const std::vector<int> &block_odd_axis)
```
