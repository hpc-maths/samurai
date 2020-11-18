Algorithm
=========

Apply a function on all the intervals.

The function must have the following format `f(level, interval, index[dim-1])`
where `interval` is the interval in the x-direction and `index[dim-1]`
is an array with the index for the other dimensions.

Example:

.. code-block:: c++

    for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index)
    {
        auto i = interval;
        auto j = index[0];
        field(level, i, j) = field(level + 1, 2 * i, 2 *j);
    });

