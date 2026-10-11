// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <iostream>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/operators_base.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

template <std::size_t dim, class TInterval>
class my_projection_op : public samurai::field_operator_base<dim, TInterval>
{
  public:

    INIT_OPERATOR(my_projection_op)

    template <class T>
    SAMURAI_INLINE void operator()(samurai::Dim<1>, T& field) const
    {
        field(level, i) = .5 * (field(level + 1, 2 * i) + field(level + 1, 2 * i + 1));
    }

    template <class T>
    SAMURAI_INLINE void operator()(samurai::Dim<2>, T& field) const
    {
        field(level, i, j) = .25
                           * (field(level + 1, 2 * i, 2 * j) + field(level + 1, 2 * i, 2 * j + 1) + field(level + 1, 2 * i + 1, 2 * j)
                              + field(level + 1, 2 * i + 1, 2 * j + 1));
    }

    template <class T>
    SAMURAI_INLINE void operator()(samurai::Dim<3>, T& field) const
    {
        field(level, i, j, k) = .125
                              * (field(level + 1, 2 * i, 2 * j, 2 * k) + field(level + 1, 2 * i + 1, 2 * j, 2 * k)
                                 + field(level + 1, 2 * i, 2 * j + 1, 2 * k) + field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)
                                 + field(level + 1, 2 * i, 2 * j, 2 * k + 1) + field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)
                                 + field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) + field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1));
    }
};

template <class T>
SAMURAI_INLINE auto my_projection(T&& field)
{
    return samurai::make_field_operator_function<my_projection_op>(std::forward<T>(field));
}

// Checks my_projection in 2D and 3D, defined after main
template <std::size_t dim>
void print_mean_of_children();

int main()
{
    samurai::initialize();

    constexpr std::size_t dim = 1;
    samurai::CellList<dim> cl;
    samurai::CellArray<dim> ca;

    cl[0][{}].add_interval({0, 10});
    cl[1][{}].add_interval({2, 6});
    cl[1][{}].add_interval({11, 15});

    ca = {cl, true};

    std::cout << ca << "\n";

    auto subset = samurai::intersection(ca[0], ca[1]);

    subset(
        [&](const auto& i, auto)
        {
            std::cout << "intersection found in " << i << std::endl;
        });

    subset.on(0)(
        [&](const auto& i, auto)
        {
            std::cout << "intersection found in " << i << std::endl;
        });

    subset.on(3)(
        [&](const auto& i, auto)
        {
            std::cout << "intersection found in " << i << std::endl;
        });

    auto subset_d = samurai::difference(ca[0], ca[1]);
    subset_d(
        [&](const auto& i, auto)
        {
            std::cout << "difference found in " << i << std::endl;
        });

    auto u = samurai::make_scalar_field<double>("u", ca);
    u.fill(0);
    samurai::for_each_cell(ca[1],
                           [&](auto cell)
                           {
                               u[cell] = cell.indices[0];
                           });

    auto subset1 = samurai::intersection(ca[0], samurai::contract(ca[1], 1));
    subset1.on(0)(
        [&](const auto& i, auto)
        {
            u(0, i) = 0.5 * (u(1, 2 * i) + u(1, 2 * i + 1));
        });

    std::cout << u << "\n";

    subset1.on(0).apply_op(my_projection(u));

    std::cout << u << "\n";

    print_mean_of_children<2>();
    print_mean_of_children<3>();

    samurai::finalize();
    return 0;
}

// Apply my_projection on a coarse cell whose 2^dim children hold 1 to 2^dim,
// and print the value it gives to the coarse cell next to their mean, (2^dim + 1) / 2.
template <std::size_t dim>
void print_mean_of_children()
{
    using index_yz_t = typename samurai::LevelCellList<dim>::index_yz_t;

    samurai::CellList<dim> cl;
    cl[0][index_yz_t{}].add_interval({0, 1});
    for (std::size_t c = 0; c < (std::size_t{1} << (dim - 1)); ++c)
    {
        index_yz_t index;
        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            index[d] = static_cast<int>((c >> d) & 1);
        }
        cl[1][index].add_interval({0, 2});
    }
    samurai::CellArray<dim> ca = {cl, true};

    auto u = samurai::make_scalar_field<double>("u", ca);
    u.fill(0);
    samurai::for_each_cell(ca[1],
                           [&](auto cell)
                           {
                               double child = 1;
                               for (std::size_t d = 0; d < dim; ++d)
                               {
                                   child += static_cast<double>(cell.indices[d] << d);
                               }
                               u[cell] = child;
                           });

    samurai::intersection(ca[0], ca[1]).on(0).apply_op(my_projection(u));

    const std::size_t n_children = std::size_t{1} << dim;
    samurai::for_each_cell(ca[0],
                           [&](auto cell)
                           {
                               std::cout << dim << "D: my_projection gives " << u[cell] << ", the mean of the " << n_children
                                         << " children is " << static_cast<double>(n_children + 1) / 2 << std::endl;
                           });
}
