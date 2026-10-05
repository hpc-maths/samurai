# Timers

This page documents the timers of {{ project }}.
`samurai::Timers` is a registry of named timers: a timer started while another one runs becomes its child, and `print()` writes the hierarchy with the time, the number of calls and the number of processed cells of each timer.
`samurai::times::timers` is the registry that {{ project }} and `samurai::ScopedTimer` use; `samurai::ScopedTimer` starts a timer when it is constructed and stops it when it goes out of scope.
`samurai::initialize` enables the registry when the program runs with `--timers`, and `samurai::finalize` prints the report.
Include `samurai/timers.hpp` to get all of them.
For how to time your own code, see the {doc}`timers how-to <../howto/timers>`.

## Scoped timer

```{doxygenclass} samurai::ScopedTimer
:members: ScopedTimer, set_cells, ~ScopedTimer
:undoc-members:
```

## Timer registry

```{doxygenvariable} samurai::times::timers
```

```{doxygenclass} samurai::Timers
:members:
:undoc-members:
```
