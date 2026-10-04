# Multiresolution adaptation

This page documents the multiresolution adaptation of {{ project }}.
`samurai::make_MRAdapt` takes the fields that drive the adaptation and returns a `samurai::Adapt` object; calling this object with a `samurai::mra_config` computes the details of the fields, tags the cells to keep, refine or coarsen, rebuilds the mesh and transfers the fields onto it.
`samurai::mra_config` holds the threshold `epsilon`, the `regularity` and the choice of relative details.
Include `samurai/mr/adapt.hpp` to get all of them.
For how to adapt a mesh in a time loop, see the {doc}`adaptation how-to <../howto/adapt>`; for the mesh itself, see {doc}`mesh`.

## Adaptation criterion

`samurai::mra_config` stores three values, set with chained setters and read with the getters of the same name:

| Setter | Default | Command-line option |
| --- | --- | --- |
| `epsilon(double)` | `1e-4` | `--mr-eps` |
| `regularity(double)` | `1` | `--mr-reg` |
| `relative_detail(bool)` | `false` | `--mr-rel-detail` |

Each call of the adaptation calls `parse_args()` on the configuration, so the command-line options replace the values set in the code.

```{doxygenclass} samurai::mra_config
:members:
:undoc-members:
```

## Create the adaptation

`make_MRAdapt` takes one or more fields on the same mesh, optionally preceded by a prediction function.
Without a prediction function, the adaptation transfers the fields with the prediction of the order set by the `prediction_stencil_radius` of the mesh configuration.
The overloads with a `bool` template argument set the `enlarge_` argument of `samurai::Adapt`, which the adaptation does not read.

```{doxygenfile} adapt.hpp
:sections: innernamespace func
```

`keep_boundary_refined` tags the cells of the maximum level along the domain boundary as kept.
The adaptation calls it when the program runs with `--refine-boundary`.

## Run the adaptation

A call to the `samurai::Adapt` object repeats the tagging and the mesh update up to `max_level - min_level` times, and stops as soon as the mesh no longer changes.
The fields passed after the configuration follow the new mesh without driving the adaptation.
The overload that takes `eps` and `regularity` as numbers is deprecated.

```{doxygenclass} samurai::Adapt
:members:
:undoc-members:
```
