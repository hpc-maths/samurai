Philosophy of |project|
=======================

The main goal of |project| is to provide a data structure adapted to adaptive mesh refinement methods based on cartesian grid. Most of the data structures used for these kinds of applications are based on the construction of a tree (quadtree in 2d or octree in 3d). The main advantage when you use a tree data structure is that you can add or delete nodes in :math:`O(1)` if you know where they are or where to add them. But if you want to search for a given node inside the tree, it can take a lot of time. To improve this search time, you can use a space-filling curve to numbering your nodes in a certain way (see p4est_ for example). This allows being sure that the neighborhood of a given node is always near this node into the data structure. It's really important because you probably want to solve a PDE using numerical methods such as finite volume, finite element methods on your adaptive mesh. This means that you have a stencil and for each cell of your mesh you have to find its neighborhood to apply your operator. So you want to make it fast.

The following figure tries to highlight the issue of the locality



.. _p4est: http://www.p4est.org/
