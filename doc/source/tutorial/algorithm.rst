Algorithm
=========

In |project|, two different algorithms are implemented to browse all the cells of the mesh. The choice depends if you have a local operator or an operator with a stencil.

If the operator is local, we can use :cpp:func:`samurai::for_each_cell`, :cpp:func:`samurai::for_each_interval` otherwise.

.. note::

    :cpp:func:`samurai::for_each_interval` can be used easily in every situations.

Apply a function on all cells
-----------------------------

We saw in the previous section that we can access the data field using :cpp:class:`Cell`. The :cpp:func:`samurai::for_each_cell` algorithm browses all the cells of the mesh and applies a lambda function on it. It can be useful when we want to initialize the field.

Let's give an example

.. code-block:: c++

    auto u = samurai::make_field<double, 1>("my_field", mesh);

    samurai::for_each_cell(mesh, [&](auto cell)
    {
        auto x = cell.center(0);
        auto y = cell.center(1);

        u[cell] = cos(x)*sin(y);
    });

The first parameter is the mesh where we want to browse all the cells and the second parameter is a lambda function with one parameter: the cell.

Apply a function using intervals
--------------------------------

We can also apply a function on a given interval using :cpp:func:`samurai::for_each_interval` as illustrated in the following example

.. code-block:: c++

    auto u = samurai::make_field<double, 1>("my_field", mesh);

    samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index)
    {
        auto i = interval;
        auto j = index[0];
        u(level, i, j) = i;
    });

The first parameter is the mesh where we want to browse all the intervals and the second parameter is a lambda function. Tis lambda function has three parameters: the level of the interval, the interval in the x-direction, and `index[dim-1]`
an array with the index for the other dimensions.

