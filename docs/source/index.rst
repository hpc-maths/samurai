========================
 |project| documentation
========================

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorial <tutorial/index>
   reference <reference/index>
   api <api/index>

The main goal of |project| is to provide a new data structure based on intervals and algebra of sets to handle efficiently adaptive mesh refinement methods based on a cartesian grid. Such an approach has to be versatile enough to handle both AMR and multi-resolution methods.

Most of the existing data structures used for such a purpose are based on the construction of a tree (quadtree in 2d or octree in 3d).

The main advantage when using a tree data structure is that adding or deleting nodes can be achieved in :math:`O(1)` once it is known where they are or where to add them.
But searching for a given node inside the tree can take a lot of time. To improve this search time, one usual way is to use a space-filling curve to locate your nodes in a linear way (see p4est_ for example relying on z-curve - Courbe de Lebesgue in French). Such an approach can guarantee that the neighborhood of a given node is always near this node inside the data structure. It is especially important for the resolution of PDEs using numerical methods such as finite volume, finite element methods on the adaptive mesh.

This means that, based on a stencil and for each cell of your mesh, a discretized operator has to evaluate various neighborhood cells or potentially reconstruct them for the tree structure, and that has to be done fast.
In order to improve the efficiency of the tree / space-filling curve, the proposed new data structure relies on intervals at various levels of the mesh and most of the operations on the mesh are conducted through operation in an algebra of sets / subsets of the mesh. The approach aims at combining three objectives :

#. improve the efficiency of the mesh handling compared to the combination tree / space-filling curves,
#. decouple the mesh evolution from the numerical methods used in order to resolve the PDEs and make it especially intuitive to implement a new scheme,
#. introduce an approach which will have good properties for distributed parallel computing.

SAMURAI: Structured Adaptive mesh and MUlti-Resolution based on Algebra of Intervals.

.. _p4est: http://www.p4est.org/
