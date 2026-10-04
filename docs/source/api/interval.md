# Interval struct

`samurai::Interval` is the half-open run `[start, end)` of integer coordinates along x at one level, the unit from which {{ project }} builds its meshes.
Its `index` member gives where the values of the interval live in the storage of a field: the value of the cell at coordinate `i` is at position `index + i`.
The struct also gives the arithmetic used to move an interval in space (`+`, `-`), to scale it (`*`, `/`) and to change its level (`>>`, `<<`).
For a guided introduction to intervals and how they describe a mesh, see the {doc}`interval tutorial <../tutorial/interval>`.

```{doxygenclass} samurai::Interval
:members:
```
