#pragma once
#include <array>
#include <utility>

#include "math.hpp"

namespace mure
{
    enum class BCType
    {
        dirichlet = 0,
        neumann = 1,
        periodic = 2
    };

    template<std::size_t Dim>
    struct BC
    {
        std::vector<std::pair<BCType, double>> type;
    };
}