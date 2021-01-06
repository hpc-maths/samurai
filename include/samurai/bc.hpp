// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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