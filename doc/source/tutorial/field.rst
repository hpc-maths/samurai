Define and use a field
======================

As we saw in the previous part (:doc:`interval`), a :cpp:class:`samurai::CellArray` defines an index for the intervals in the x-direction to link the mesh and the data storage of the problem solution.

The construction of a field is made using a :cpp:class:`samurai::CellArray` or a derived class from :cpp:class:`samurai::Mesh`.

The example below shows how to initialize a field with 2 elements by cells of type `double`.

.. code-block:: c++

    auto u = samurai::make_field<double, 2>("field_name", mesh);

The name of the field is used when we want to save the solution in hdf5 format.

A field is accessible in several ways in |project|.

The first one is using a :cpp:class:`samurai::Cell` as in this example

.. code-block:: c++

    samurai::Cell<coord_index_t, dim> cell{level, indices, index};

    u[cell] = ...;

:cpp:class:`samurai::Cell` is defined by the level, the coordinates of the cell using integer and the index where we can find this cell into the field. In general, we don't have to initialize a cell. We can use algorithms that perform a loop over the cells of the mesh as we will see in the next section (:doc:`algorithm`).

The second way is to access the elements of the field as if we were on a cartesian grid. Let's give an example to better understand how it works

.. code-block:: c++

    u(level, i, j) = ...

If we omit the level attribute, we observe that we can access the data of the field by using the indices `i`, `j`, `k` as we would make in a uniform structured code. The level is needed here to know where the indices live. Let's recall that `i=1` at level `0` is completely different from `i=1` at level`10`. The other difference is that the parameter `i` is not a scalar but an interval. On the other hand, the other indices (`j`, `k`, ...) are scalars.

`u(level, i, j)` returns a `xtensor view <https://xtensor.readthedocs.io/en/latest/view.html>`_ of the field where are stored the values for the given parameters. It means that we can use lazy expression as in `xtensor <https://xtensor.readthedocs.io>`_ to update the data. For example

.. code-block:: c++

    double dx = 1./(1<<level);
    auto x = dx * xt::arange(i.start, i.end);
    auto y = dx * j;
    u(level, i, j) = xt::cos(x)*xt::sin(y);

In this example, `x` is a vector of the size of the interval and `y` is a scalar. Remember that the field `u` has two components, this expression is applied to both of them.

If we want to apply this expression only to one component, we can add a parameter at the beginning to specify which one

.. code-block:: c++

    u(1, level, i, j) = xt::cos(x)*xt::sin(y);