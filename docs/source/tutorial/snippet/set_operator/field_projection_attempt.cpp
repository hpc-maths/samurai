// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The field projection on the intersection of the two levels of
// demos/tutorial/set_operator.cpp, without contraction: it reads fine cells
// that level 1 does not hold, and throws std::out_of_range.

#include <iostream>
#include <stdexcept>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

int main()
{
    samurai::initialize();

    constexpr std::size_t dim = 1;
    samurai::CellList<dim> cl;
    cl[0][{}].add_interval({0, 10});
    cl[1][{}].add_interval({2, 6});
    cl[1][{}].add_interval({11, 15});
    samurai::CellArray<dim> ca = {cl, true};

    auto u = samurai::make_scalar_field<double>("u", ca);
    u.fill(0);
    samurai::for_each_cell(ca[1],
                           [&](auto cell)
                           {
                               u[cell] = cell.indices[0];
                           });

    // The interval of level 0 in progress, named if a read throws. The message
    // of the exception is not printed: it depends on which of the two reads
    // fails first, and C++ leaves that order to the compiler.
    samurai::CellArray<dim>::interval_t interval;
    try
    {
        auto set = samurai::intersection(ca[0], ca[1]).on(0);
        set(
            [&](const auto& i, auto)
            {
                interval = i;
                std::cout << "field projection on " << i << std::endl;
                u(0, i) = 0.5 * (u(1, 2 * i) + u(1, 2 * i + 1));
            });
    }
    catch (const std::out_of_range&)
    {
        std::cout << "std::out_of_range thrown by the field projection on "
                  << interval << std::endl;
        samurai::finalize();
        return 1;
    }

    samurai::finalize();
    return 0;
}
