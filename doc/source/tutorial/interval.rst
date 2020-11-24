Interval and cartesian grid representation
==========================================

Introduction
------------

A cartesian grid is composed of several cells with a given length. The length can be the same for all the cells such as in the case of uniform grids. A cartesian grid can be represented by intervals. To illustrate this idea, we start with a simple 1D example

.. image:: ./figures/segments.png
    :width: 80%
    :align: center

The domain can be defined by the interval :math:`[0, 13]`. We observe that this domain is composed of five cells which can also be defined as intervals

- cell 1: :math:`[0, 4]`
- cell 2: :math:`[4, 5]`
- cell 3: :math:`[5, 7]`
- cell 4: :math:`[7, 9]`
- cell 5: :math:`[9, 13]`

We also observe that several cells have the same width and then they can be regrouped by width (or resolution)

- width of size 1: :math:`[4, 5]`
- width of size 2: :math:`[5, 9]`
- width of size 4: :math:`[0, 4]`, :math:`[9, 13]`

Since we know what the resolution is, we can regroup the cells to construct contiguous intervals. For the cells with a width of size 2, we have two cells: :math:`[5, 7]` and :math:`[7, 9]` which forms the interval :math:`[5, 9]`. And, since we know the resolution of this given interval, the two cells can be reconstructed.

If we plot the initial domain using the different levels of resolution, we have

.. image:: ./figures/segments-resolution.png
    :width: 80%
    :align: center

In this example, we choose to not have intersections between the resolution levels but it is not mandatory and we can imagine a domain with overlapping regions

.. image:: ./figures/segments-resolution-overlap.png
    :width: 80%
    :align: center

.. note::

    The construction of the cells in |project| has some constraints and it will be not possible to make exactly the domain describes previously. These constraints will be explained in the next section.

Interval definition
-------------------

The data structure widely used by |project| is an interval. An interval is described as follows

.. image:: ./figures/interval.png
    :width: 200
    :align: center

The interval is defined by its start and end values. And we introduce two new properties

- the step which is used to browse the interval with a given step,
- the index which is used as an offset on other data structures (more details will be given in the following).

.. warning::

    - In the introduction, we didn't pay attention to the type of bounds describing the interval. In |project|, the start and the end of the interval are integers.

    - It is also important to notice that the end value of the interval is not included.

As in the introduction, we have multiple resolutions which mean different cell sizes. The grid resolution is defined by a level. The size of a cell is fixed by its level

.. math:: \Delta x = \frac{1}{2^{level}}.
    :label: dx

The size is the same in all the directions and, therefore, the cells are represented by squares in 2D and cubes in 3D.

In |project|, intervals are represented by integers and not by real numbers as in the introduction. This is because an interval is defined by the cells of a given level. The figure below illustrates the idea

.. image:: ./figures/interval-2.png
    :width: 250
    :align: center

We have two cells :math:`0` and :math:`1` on the level :math:`l`. Therefore, we can write that the interval :math:`[0, 2[` describes this domain at the level :math:`l`.

.. note::

    We could have chosen to describe the interval as :math:`[0, 1]` but the choice to take the end of the interval not included is important for the algebra of intervals (see :ref:`AlgebraOfSet`).

.. _cell:

Cell properties
---------------

Given an interval and a level, we can reconstruct the cells.

- The number of cells contained in an interval is equal to the size of the interval.
- The size of an interval is given by

.. math::
    size = end - start.

- The coordinates of the minimum corner of a cell is given by

.. math::

    corner = (i \Delta x, j \Delta x, ...)

where :math:`\Delta x` is given by equation :eq:`dx`.

- The coordinates of the center of a cell is given by

.. math::

    center = \left(\left(i + \frac{1}{2}\right) \Delta x, \left(j + \frac{1}{2}\right) \Delta x, ...\right).

Constraints on the grid representation
--------------------------------------

Since the grid is constructed by a set of intervals defined by the cell numbering on a given cell, the possible coordinates are fixed. Suppose we are at level 1 with :math:`\Delta x = 0.5` for a 1D problem, it is therefore impossible by definition to have a cell with a center equal to :math:`\frac{1}{3}`.

.. note::

    We can imagine in the next versions to have an operator which transforms the grid represented by intervals on cells to the real domain.

The other constraint is that a cell at the level :math:`l` is included in a cell at the lowest level.

.. image:: ./figures/interval-3.png
    :width: 80%
    :align: center

.. note::

    this property is important for mesh adaptation.

1D mesh example
---------------

Let's now take a 1D example with various levels and explain how the mesh is stored in |project|.

.. image:: ./figures/interval_example_1D.png
    :width: 80%
    :align: center

For each level, the intervals are

- level 0: :math:`[0, 2[`, :math:`[5, 6[`
- level 1: :math:`[4, 7[`, :math:`[8, 10[`
- level 2: :math:`[14, 16[`

The real intervals are given by the level and :math:`\Delta x` defined by :eq:`dx`

- level 0: :math:`[0., 2.]`, :math:`[5., 6.]`
- level 1: :math:`[3., 3.5]`, :math:`[4., 5.]`
- level 2: :math:`[3.5, 4.]`

.. note::

    There are no overlapping regions in this example. It is a choice to make the example more readable. We will see in the tutorials mesh constructions with overlap. It is often the case when mesh adaptation are performed and ghost cells are needed to update the solution using a spatial operator with a stencil.

The following code implements this example using |project|.

.. literalinclude:: snippet/interval.cpp
  :language: c++

The output is

.. literalinclude:: snippet/interval_output.txt

The computation of the index values represented by the `@` operator will be explained in a next section.

Two new data structures are used in this example :cpp:class:`samurai::CellList` and :cpp:class:`samurai::CellArray` which are c++ arrays of size `max_level` defined as a template parameter. The default size is 16.

:cpp:class:`samurai::CellList` is used to efficiently add new intervals when the mesh is constructed. As its name suggest, :cpp:class:`samurai::CellList` is nothing more than a list of intervals in the x-direction. This list is stored in a map where the keys are the index in the other dimensions (y, z, ...) and the values are the list of intervals in the x-direction.

.. note::

    We will give an example of a 2D case in the following to better explain how are constructed the keys. For 1D problem the key is empty. This is why we use :cpp_code:`{}` in the construction of the :cpp:class:`samurai::CellList` like in :cpp_code:`cl[0][{}].add_interval({ 0,  2});`

:cpp:class:`samurai::CellList` data structure is efficient to add new elements (search, removal and insertion operations have logarithmic complexity for :cpp_code:`std::map`). But when scientific computing algorithms such as numerical schemes on a stencil must be applied, it is important to loop over the cells efficiently without a search algorithm. We also want to apply algebra of intervals for a given dimension which means that a dimension should have its own representation by intervals in a compact writing.

:cpp:class:`samurai::CellArray` is precisely used to compress the representation of the mesh where each dimension has its own interval list stored as an array.

In our 1D example, the :cpp:class:`samurai::CellList` associated with this mesh is

.. code::

    level 0:
        x: [0, 2[, [5, 6[

    level 1:
        x: [4, 7[, [8, 10[

    level 2:
        x: [14, 16[

the :cpp:class:`samurai::CellArray` is defined in the same way. The only difference is that `x` is a :cpp_code:`std::forward_list` of :cpp:class:`samurai::Interval` in :cpp:class:`samurai::CellList` and a :cpp_code:`std::vector` of :cpp:class:`samurai::Interval` in :cpp:class:`samurai::CellArray`.

To understand the differences, we have to give an example in 2D.

2D mesh example
---------------

The example below will help to better understand the difference between :cpp:class:`samurai::CellList` and :cpp:class:`samurai::CellArray`.

.. image:: ./figures/2D_mesh.png
    :width: 60%
    :align: center

The :cpp:class:`samurai::CellList` associated with this mesh is

.. code::

    level 0:
        y: 0
            x: [0, 4[
        y: 1
            x: [0, 1[, [3, 4[
        y: 2
            x: [0, 1[, [3, 4[
        y: 3
            x: [0, 3[

    level 1:
        y: 2
            x: [2, 6[
        y: 3
            x: [2, 6[
        y: 4
            x: [2, 4[, [5, 6[
        y: 5
            x: [2, 6[
        y: 6
            x: [6, 8[
        y: 7
            x: [6, 7[

    level 2:
        y: 8
            x: [8, 10[
        y: 9
            x: [8, 10[
        y: 14
            x: [14, 16[
        y: 15
            x: [14, 16[

The key of the map in :cpp:class:`samurai::CellList` is the index in `y` and the value of the key is the list of intervals in the x-direction for this index.

The :cpp:class:`samurai::CellArray` is

.. code::

    level 0:
        x: [0, 4[, [0, 1[, [3, 4[, [0, 1[, [3, 4[, [0, 3[
        y: [0, 4[@0
        y-offset: [0, 1, 3, 5, 6]

    level 1:
        x: [2, 6[, [2, 6[, [2, 4[, [5, 6[, [2, 6[, [6, 8[, [6, 7[
        y: [2, 8[@-2
        y-offset: [0, 1, 3, 5, 6, 7, 8]

    level 2:
        x: [8, 10[, [8, 10[, [14, 16[, [14, 16[
        y: [8, 10[@-8, [14, 16[@-12
        y-offset: [0, 1, 2, 3, 4]

How the :cpp:class:`samurai::CellArray` is constructed from the :cpp:class:`samurai::CellList` ?

First, we concatenate the intervals in the x-direction for each index `y` for a given level. Therefore, the x array at level 2 representing the intervals in the x-direction is

.. code::

    x: [8, 10[, [8, 10[, [14, 16[, [14, 16[

Then, we try to construct intervals in the y-direction from the keys. For level 2, we have `y = 8, 9, 14, 15`. Therefore, we can construct two intervals: :math:`[8, 10[` and :math:`[14, 16[`.

The compressed view of the :cpp:class:`samurai::CellList` at level 2 is as follows

.. code::

    x: [8, 10[, [8, 10[, [14, 16[, [14, 16[
    y: [8, 10[, [14, 16[

Now, we have to connect each y entry to the corresponding intervals in the x-direction. We use for that a new array called `y-offset` and the index of the interval represented by the operator `@`.

Each y has one interval in the x-direction. The `y-offset` indicates for each y where are the corresponding intervals in the x-direction in the array x.

- for `y = 8`, there is one interval in x-direction,
- for `y = 9`, there is one interval in x-direction,
- for `y = 14`, there is one interval in x-direction,
- for `y = 15`, there is one interval in x-direction.

The `y-offset` is the array `[0, 1, 2, 3, 4]`. The size of this array is the number of y + 1 and indicates that :math:`y[i]` has the intervals in the x-direction between :math:`y-offset[i]` and :math:`y-offset[i+1]` in the x array.

One point remains to be clarified: how many elements y have we already gone through to know where to look in the `y-offsets`? This is where the index plays an important role. If we look at the interval y :math:`[14, 16[`, we know that the corresponding index in `y-offsets` for `y = 14` is `y-offset[2]`. To obtain the right index, we choose the index defined in the interval by the `@` operator in order to have `y + index` is equal to the entry in the `y-offset` entry. Then for `y = 14`, if the index is equal to `-12`, we find `y-offset[y + @index] = y-offset[14 - 12] = y-offset[2]`.

If the same operation is made to compute the `y-offset` and the index on each interval in the y-direction, we find the :cpp:class:`samurai::CellArray`

.. code::

    level 0:
        x: [0, 4[, [0, 1[, [3, 4[, [0, 1[, [3, 4[, [0, 3[
        y: [0, 4[@0
        y-offset: [0, 1, 3, 5, 6]

    level 1:
        x: [2, 6[, [2, 6[, [2, 4[, [5, 6[, [2, 6[, [6, 8[, [6, 7[
        y: [2, 8[@-2
        y-offset: [0, 1, 3, 5, 6, 7, 8]

    level 2:
        x: [8, 10[, [8, 10[, [14, 16[, [14, 16[
        y: [8, 10[@-8, [14, 16[@-12
        y-offset: [0, 1, 2, 3, 4]

.. note::

    The algorithm described here to compress the :cpp:class:`samurai::CellList` into the :cpp:class:`samurai::CellArray` for the 2D is a recursive algorithm and, thus, it can be easily used for other dimensions.

    We are not limited to 1D, 2D, and 3D problems. We can construct a grid in high dimensions.

The implementation of this example is

.. literalinclude:: snippet/2d_mesh_representation.cpp
  :language: c++

And the output is

.. literalinclude:: snippet/2d_mesh_representation_output.txt

Build a grid from a box
-----------------------

Since the beginning, we have used :cpp:class:`samurai::CellList` to build :cpp:class:`samurai::CellArray`. We also can easily initialize a :cpp:class:`samurai::CellArray` at a given level with a uniform cartesian grid by defining a box.

- The box can be 1D, 2D, or 3D.
- The box can define the cells involved in the grid using integers or a box in real coordinates.

The following example uses a box in real coordinates

.. literalinclude:: snippet/2d_mesh_box.cpp
  :language: c++

The box is defined by its minimum and maximum corners. In this example, the box is therefore :math:`[-1, 1] \times [-1, 1]`. The space step is chosen from the given level which means :math:`\Delta x = 2^{-3} = 0.125`. The number of cells is defined by the length of the box and the space step.

.. literalinclude:: snippet/2d_mesh_box_output.txt

We obtain the following mesh

.. image:: ./figures/2D_mesh_box.png
    :width: 60%
    :align: center

.. warning::

    Since the size of the cells is fixed at a given level and their coordinates are fixed by their integer representation, it is not always possible to build a box in real coordinates where the bounds of the box correspond to a corner point of a cell.