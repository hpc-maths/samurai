# Cell struct

`samurai::Cell` is the hypercube identified by a level and integer coordinates, together with its index in the storage of a field.
{{ project }} does not store cells: the loops build a `Cell` from an interval when they visit it, for example `samurai::for_each_cell`.
The struct gives the corner, the center and the face centers of the cell in real coordinates, and its length.
For how to loop over the cells of a mesh, see the {doc}`loop how-to <../howto/loop>`; for the relation between a cell and its interval, see the {ref}`cell properties <cell>` in the interval tutorial.

```{doxygenclass} samurai::Cell
:members:
```
