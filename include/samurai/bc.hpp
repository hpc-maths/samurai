#pragma once
#include <vector>
#include <utility>

namespace samurai
{
    enum class BCType
    {
        dirichlet = 0,
        neumann = 1,
        periodic = 2,
        interpolation = 3 // Reconstruct the function by linear approximation
    };

    template<std::size_t Dim>
    struct BC
    {
        std::vector<std::pair<BCType, double>> type;
    };
} // namespace samurai