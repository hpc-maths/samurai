# How-to: loop over cells in a samurai mesh

In this how-to guide, we will show you how to loop over the cells of a samurai mesh. Looping over cells is a common operation when you want to perform computations or apply algorithms on each cell of the mesh.

To follow this guide, you should already have a samurai mesh created. If you don't know how to create a mesh, please refer to the mesh how-to guide [here](mesh.md).

It can be also interesting to have a samurai field defined on the mesh. If you don't know how to create a field, please refer to the field how-to guide [here](field.md).

## Looping over cells

To loop over the cells of a samurai mesh, you can use the `for_each_cell` function provided by samurai. Here is a simple example of how to loop over the cells of a multi-resolution mesh and print the level and index of each cell:

```cpp
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/algorithm.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t = samurai::MRConfig<dim>;

    samurai::Box<dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::MRMesh<config_t> mesh(box, 2, 5); // min level 2, max level 5

    samurai::for_each_cell(mesh, [&](const auto& cell)
    {
        std::cout << "Cell level: " << cell.level() << ", center: " << cell.center() << std::endl;
    });

    return 0;
}
```

In this example, we first create a 2D multi-resolution mesh over the box defined from $(0.0, 0.0)$ to $(1.0, 1.0)$ with a minimum refinement level of 2 and a maximum refinement level of 5. Then, we use the `for_each_cell` function to loop over each cell in the mesh. Inside the loop, we print the level and center of each cell.

You can also access and modify the values of a samurai field while looping over the cells. Here is an example of how to set the values of a scalar field to the level of each cell:

```cpp
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/algorithm.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t = samurai::MRConfig<dim>;

    samurai::Box<dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::MRMesh<config_t> mesh(box, 2, 5); // min level 2, max level 5

    auto field = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_cell(mesh, [&](const auto& cell)
    {
        auto x = cell.center(0);
        auto y = cell.center(1);

        field[cell] = std::exp(-( (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) ) * 20.0);
    });

    return 0;
}
```

In this example, we create a scalar field named "u" on the multi-resolution mesh. Inside the loop, we compute the center coordinates of each cell and set the value of the field at that cell to a Gaussian function centered at (0.5, 0.5).

## Looping over intervals

Another way to iterate over the cells of a samurai mesh is to use intervals. You can use the `for_each_interval` function to loop over intervals. Here is an example:

```cpp
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/algorithm.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t = samurai::MRConfig<dim>;

    samurai::Box<dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::MRMesh<config_t> mesh(box, 2, 5); // min level 2, max level 5

    samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index)
    {
        auto y = index[0];
        std::cout << "Level: " << level << ", x: " << interval << ", y: " << y << std::endl;
    });

    return 0;
}
```

In this example, we use the `for_each_interval` function to loop over each interval in the mesh. Inside the loop, we print the interval, level, and index of each interval. `index` is an array of size $dim-1$ that contains the indices of the other dimensions ($y$, $z$, ...). If you want to deeper understand intervals, please refer to the interval tutorial [here](../tutorial/interval.rst)

Let's see how to set the values of a scalar field using intervals:

```cpp
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/algorithm.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t = samurai::MRConfig<dim>;

    samurai::Box<dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::MRMesh<config_t> mesh(box, 2, 5); // min level 2, max level 5

    auto field = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index)
    {
        auto y = index[0];
        auto x = mesh.cell_length() * xt::arange(interval.start, interval.end());
        field(level, i, index)  = xt::exp(-( (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) ) * 20.0);
    });

    return 0;
}
```

