// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai::load_balancing::weight
{
    /**
     * Uniform weight policy: every cell costs 1.
     *
     * A weight policy is any callable `double(const cell_t&)`; it abstracts the
     * computational cost of a cell so that strategies balance *work* instead of
     * cell counts. Further policies (`per_level`, `from_field`) are introduced
     * by step 2 of the load balancing roadmap (docs/load_balancing_roadmap.md).
     */
    inline auto uniform()
    {
        return [](const auto& /*cell*/)
        {
            return 1.0;
        };
    }
}
