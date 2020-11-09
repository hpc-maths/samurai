Graduation example: case 2
==========================

In this tutorial, the mesh is constituted of cells at different levels with overlap and we want a graduated mesh at the end. The complete example can be downloaded here: :download:`graduation case 2 <../../../demos/tutorial/graduation_case_2.cpp>`

First, we need an initial mesh with overlap between levels. We will generate it randomly in the 2D domain :math:`[0, 1] \times [0, 1]`. The idea is to randomly add cells at different levels. The implementation of the initial mesh is described in the following code.

.. code-block:: c++

    auto generate_mesh(std::size_t min_level, std::size_t max_level, std::size_t nsamples = 100)
    {
        constexpr std::size_t dim = 2;

        mure::CellList<dim> cl;
        cl[0][{0}].add_point(0);

        for(std::size_t s = 0; s < nsamples; ++s)
        {
            auto level = std::experimental::randint(min_level, max_level);
            auto x = std::experimental::randint(0, (1<<level) - 1);
            auto y = std::experimental::randint(0, (1<<level) - 1);

            cl[level][{y}].add_point(x);
        }

        return mure::CellArray<dim>(cl, true);
    }

Let's explain step by step this function. There are three parameters: `min_level` is the minimum level where a cell can be added, `max_level` is the maximum level where a cell can be added, and `nsamples` is the number of cells which will be randomly added to the final mesh.


We first create a `CellList` to add new cells or intervals efficiently. We add into it the cell `{0, 0}` at level `0` which corresponds to the square :math:`[0, 1] \times [0, 1]` to be sure that we have at the end the entire domain :math:`[0, 1] \times [0, 1]`.

.. code-block:: c++

    mure::CellList<dim> cl;
    cl[0][{0}].add_point(0);

And then, we create randomly `nsamples` cells.

.. code-block:: c++

    for(std::size_t s = 0; s < nsamples; ++s)
    {
        auto level = std::experimental::randint(min_level, max_level);
        auto x = std::experimental::randint(0, (1<<level) - 1);
        auto y = std::experimental::randint(0, (1<<level) - 1);

        cl[level][{y}].add_point(x);
    }

Now, we can construct the `Cellarray` from this `CellList` and return it.

.. code-block:: c++

    return mure::CellArray<dim>(cl, true);

The figure below is an example of an initial mesh with start_level = 1 and max_level = 7.

.. image:: ./figures/graduation_case_2_before.png
    :width: 80%
    :align: center

The next step is to remove all possible intersections between two levels. We will use the subset mechanism of |project| as for the previous tutorial :doc:`graduation case 1 <./graduation_case_1>`. The idea is the following: we make the intersection of the cells at a level `l` with the previous levels. If this intersection exists, then we refine the cells at the previous levels. We repeat this process until no intersections are detected.

For this algorithm, we use a field named `tag` attached to the mesh as in the previous case. This field is an array of booleans. If it is set to true, the cell must be refined, and must be kept otherwise.

The algorithm is similar to the algorithm described in :doc:`graduation case 1 <./graduation_case_1>`: only the subset definition is changed.

So, we try to find an intersection using subset construction between a level `level` and a `level_below` where `level_below < level`.

.. code-block:: c++

    auto set = mure::intersection(ca[level], ca[level_below])
              .on(level_below);

    set([&](const auto& i, const auto& index)
    {
        tag(level_below, i, index[0]) = true;
    });

And we reconstruct a new mesh using `tag` and `CellList` using the following algorithm.

.. code-block:: c++

    std::size_t min_level = ca.min_level();
    std::size_t max_level = ca.max_level();

    while(true)
    {
        auto tag = mure::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for(std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            for(std::size_t level_below = min_level; level_below < level; ++level_below)
            {
                auto set = mure::intersection(ca[level], ca[level_below]).on(level_below);
                set([&](const auto& i, const auto& index)
                {
                    tag(level_below, i, index[0]) = true;
                });
            }
        }

        mure::CellList<dim> cl;
        mure::for_each_cell(ca, [&](auto cell)
        {
            auto i = cell.indices[0];
            auto j = cell.indices[1];
            if (tag[cell])
            {
                cl[cell.level + 1][{2*j}].add_interval({2*i, 2*i+2});
                cl[cell.level + 1][{2*j + 1}].add_interval({2*i, 2*i+2});
            }
            else
            {
                cl[cell.level][{j}].add_point(i);
            }
        });
        mure::CellArray<dim> new_ca = {cl, true};

        if(new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }

The figure below is the initial mesh without intersections. The blue cells are the cells added to remove the intersections.

.. image:: ./figures/graduation_case_2_after.png
    :width: 80%
    :align: center

The graduation of this new mesh is straightforward since this is exactly the algorithm described in the previous case.

The figure below is the graduation of our initial mesh. The red cells are the cells added by the graduation.

.. image:: ./figures/graduation_case_2_after_graduated.png
    :width: 80%
    :align: center