#pragma once

#include "../../storage/containers.hpp"

namespace samurai
{
    enum class SchemeType
    {
        NonLinear,
        LinearHeterogeneous,
        LinearHomogeneous
    };

    enum class Get
    {
        Cells,
        Intervals
    };

} // end namespace samurai
