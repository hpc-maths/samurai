#pragma once
#include <samurai/algorithm.hpp>
#include <samurai/print.hpp>

template <class Mesh>
void print_mesh(Mesh& mesh)
{
    samurai::io::print("{}\n", fmt::streamed(mesh));
    samurai::for_each_cell(mesh,
                           [](const auto& cell)
                           {
                               samurai::io::print("level: {}, cell index: {}, center: {}\n", cell.level, cell.index, cell.center(0));
                           });
}

template <class Field>
bool check_nan_or_inf(const Field& f)
{
    std::size_t n      = f.mesh().nb_cells();
    bool is_nan_or_inf = false;
    for (std::size_t i = 0; i < n * Field::n_comp; ++i)
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
