#include <iostream>

#include <mure/cell_array.hpp>
#include <mure/cell_list.hpp>
#include <mure/field.hpp>

int main()
{
    constexpr std::size_t dim = 1;

    // Mesh creation
    mure::CellList<dim> cl;
    cl[0][{}].add_interval({ 0,  4});
    cl[1][{}].add_interval({ 0,  4});
    cl[1][{}].add_interval({ 6,  8});

    mure::CellArray<dim> ca{cl};

    // Initialize field u on this mesh
    auto u = mure::make_field<double, 1>("u", ca);
    mure::for_each_cell(ca, [&](auto cell)
    {
        u[cell] = cell.indices[0];
    });

    std::cout << "before projection" << std::endl;
    std::cout << u << std::endl;

    // Make projection on the intersection
    auto subset = mure::intersection(ca[0], ca[1]).on(0);
    subset([&](const auto& i, auto)
    {
        u(0, i) = 0.5*(u(1, 2 * i) + u(1, 2 * i + 1));
    });

    std::cout << "after projection" << std::endl;
    std::cout << u << std::endl;

    return 0;
}