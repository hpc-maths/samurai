# Box class

`samurai::Box` is a hyperrectangle in real coordinates, given by its minimum and maximum corners.
It describes the geometry before discretization: a mesh is built on one box, or on a domain made of several boxes and holes.
The class gives the length of the box along each direction, and the intersection and difference of two boxes.
For how to build a domain from boxes, see the {doc}`box how-to <../howto/box>`.

```{doxygenclass} samurai::Box
:members:
```
