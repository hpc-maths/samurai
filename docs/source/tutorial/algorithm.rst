Algorithm
=========

In |project|, two different algorithms are implemented to browse all the cells of the mesh in order to apply some operator on them.
The choice depends on whether one has a local operator, that is, utilizing information belonging only to one cell without interaction with the neighbors; or an operator encoding some interaction with the surrounding cells.

If the operator is local, one can use :cpp:func:`samurai::for_each_cell`, otherwise, when the operator has an extended stencil, one can employ :cpp:func:`samurai::for_each_interval`.

.. note::

    :cpp:func:`samurai::for_each_interval` can be easily used in every situation.

Apply a function on all cells
-----------------------------

In the previous section, we have seen that we can access the data field using :cpp:class:`Cell`.
The algorithm in :cpp:func:`samurai::for_each_cell` browses all the cells of the mesh and applies a lambda function on them.
It can be useful, for example, when one wants to initialize a field.

Let us give an example

.. code-block:: c++

    auto u = samurai::make_field<double, 1>("my_field", mesh);

    samurai::for_each_cell(mesh, [&](auto cell)
    {
        auto x = cell.center(0);
        auto y = cell.center(1);

        u[cell] = cos(x)*sin(y);
    });

The first parameter is the mesh we want to navigate through and the second parameter is a lambda function with one parameter: the cell.
We have used the procedure :cpp:func:`center` of the :cpp:class:`Cell` class to recover the spatial coordinates of the cell center.

Apply a function using intervals
--------------------------------

As suggested before, we can also evaluate a function on a given interval using :cpp:func:`samurai::for_each_interval` as illustrated in the following example

.. code-block:: c++

    auto u = samurai::make_field<double, 1>("my_field", mesh);

    samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index)
    {
        auto i = interval;
        auto j = index[0];
        u(level, i, j) = i;
    });

The first parameter is the mesh where we want to browse all the intervals and the second parameter is again a lambda function.
This function takes three parameters: the level of the intervals we want to pick, the interval in the x-direction, and `index[dim-1]`
an array with the index for the other dimensions.
