Algorithm Examples for the graduation of a mesh
===============================================

This tutorial will highlight three different ways to make the graduation of a mesh. A mesh is graduated when the neighbors of a cell at level :math:`l` are at most at the next or previous level. This process is important when we use adaptive mesh refinement techniques. The graduation ensures that the reconstruction of the ghost cell values of level `l` can be made with the previous or the next level.

The three cases that will be considered in the following are:

- The mesh is constituted of cells at different levels but without overlap and we want a graduated mesh at the end.
- The mesh is constituted of cells at different levels with overlap and we want a graduated mesh at the end.
- The mesh is already graduated and a mesh adaptation algorithm is performed on it. A tag array indicates which cells must be refined, kept, or coarsen to build the new mesh. We want to modify the tag array to be sure that the new mesh created from the tag will be graduated.

Case 1
------

In this case, the mesh is constituted of cells at different levels but without overlap and we want a graduated mesh at the end. First, we need an initial mesh with this property. We will generate it randomly beginning by level :math:`1` in the 2D domain :math:`[0, 1] \times [0, 1]`. The idea is to refine randomly the cells of a given level and to create a new mesh. The implementation of the initial mesh is described in the following code.

.. code-block:: c++

    auto generate_mesh(std::size_t start_level, std::size_t max_level)
    {
        constexpr std::size_t dim = 2;
        mure::Box<int, dim> box({0, 0}, {1<<start_level, 1<<start_level});
        mure::CellArray<dim> ca;

        ca[start_level] = {start_level, box};

        for(std::size_t ite = 0; ite < max_level - start_level; ++ite)
        {
            mure::CellList<dim> cl;

            mure::for_each_interval(ca, [&](std::size_t level, const auto& interval, const auto& index)
            {
                auto choice = xt::random::choice(xt::xtensor_fixed<bool, xt::xshape<2>>{true, false}, interval.size());
                for(int i = interval.start, ic = 0; i<interval.end; ++i, ++ic)
                {
                    if (choice[ic])
                    {
                        cl[level + 1][2*index].add_interval({2*i, 2*i+2});
                        cl[level + 1][2*index + 1].add_interval({2*i, 2*i+2});
                    }
                    else
                    {
                        cl[level][index].add_point(i);
                    }
                }
            });

            ca = {cl, true};
        }

        return ca;
    }

Let's explain step by step this function. There are two parameters: `start_level` is the level where we build our first 2D mesh in the domain :math:`[0, 1] \times [0, 1]` and `max_level` is the maximum level where we can have cells.

The first part is the construction of the uniform initial mesh

.. code-block:: c++

        mure::Box<int, dim> box({0, 0}, {1<<start_level, 1<<start_level});
        mure::CellArray<dim> ca;
        ca[start_level] = {start_level, box};

We construct here a `CellArray` from a box. In |project|, a box is defined by its minimum and its maximum coordinates. `CellArray` contains integers describing the mesh. The relation between the space step and the level is :math:`dx=\frac{1}{1<<level}`. We recall that our domain is :math:`[0, 1] \times [0, 1]`. Then, our box starts at :math:`[0, 0]` and needs :math:`1<<level` points to reach the maximum coordinates :math:`[0, 1]`. In the end, we assign this box to the `start_level` of the `CellArray`.

Now that we have our initial mesh, we can begin to refine it randomly. Let's start with the inner loop.

.. code-block: c++

    mure::for_each_interval(ca, [&](std::size_t level, const auto& interval, const auto& index)
    {
        auto choice = xt::random::choice(xt::xtensor_fixed<bool, xt::xshape<2>>{true, false}, interval.size());
        for(int i = interval.start, ic = 0; i<interval.end; ++i, ++ic)
        {
            if (choice[ic])
            {
                cl[level + 1][2*index].add_interval({2*i, 2*i+2});
                cl[level + 1][2*index + 1].add_interval({2*i, 2*i+2});
            }
            else
            {
                cl[level][index].add_point(i);
            }
        }
    });

Here we make a loop on the 1D intervals of each level of `ca`. The `for_each_interval` function takes a `CellArray` and a lambda function with the parameters `interval` which is the interval in the x-direction and an array `index` with the coordinates of the other dimensions. Since our domain is 2D, `index` is an array of size :math:`1` and contains the y-coordinate.

Then, from the size of the interval, we construct an xtensor container with random values `true` or `false` and make a loop over these values. If it is true, we refine our cell and we add it to a `CellList` for better performance during a construction of a mesh. If it is false, we just add this cell to the new mesh.

Our `CellList` contains the new mesh and we have now to assign it to our `CellArray`.

.. code-block: c++

    ca = {cl, true};

And we make this process `max_level - start_level` to have cells on the `max_level` at the end.

The figure below is an example of an initial mesh with `start_level = 1` and `max_level = 8`.

.. image:: ./figures/graduation_case_1_before.png
    :width: 80%
    :align: center

.. image:: ./figures/graduation_case_1_after.png
    :width: 80%
    :align: center