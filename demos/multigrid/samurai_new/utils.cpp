#pragma once
#include <samurai/algorithm.hpp>

template <class Mesh>
void print_mesh(Mesh& mesh)
{
    std::cout << mesh << std::endl;
    samurai::for_each_cell(mesh,
                           [](const auto& cell)
                           {
                               std::cout << "level: " << cell.level << ", cell index: " << cell.index << ", center: " << cell.center(0)
                                         << std::endl;
                           });
}

template <class Field>
bool check_nan_or_inf(const Field& f)
{
    std::size_t n      = f.mesh().nb_cells();
    bool is_nan_or_inf = false;
    for (std::size_t i = 0; i < n * Field::size; ++i)
    {
        double value = f.array().data()[i];
        if (std::isnan(value) || std::isinf(value) || (abs(value) < 1e-300 && abs(value) != 0))
        {
            is_nan_or_inf = true;
            break;
        }
    }
    return !is_nan_or_inf;
}
