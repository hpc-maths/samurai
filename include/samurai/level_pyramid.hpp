// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>

#include "cell_array.hpp"
#include "level_cell_array.hpp"
#include "subset/node.hpp"

namespace samurai
{
    // Static geometry (domain, corners, boundary condition regions) is stored at one
    // level and then queried at every level through a set projection,
    // self(lca).on(level). Coarsening a fine set on the fly costs 2^shift fine rows
    // per coarse row, on every traversal. A pyramid materialises that projection
    // once: pyramid[level] holds exactly the cells of self(lca).on(level), for
    // every level in [0, max_level], so the consumers read a level cell array at
    // their own level and the traversal cost stops depending on max_level - level.
    template <class CellArray, class LCA>
    void build_level_pyramid(CellArray& pyramid, const LCA& lca, std::size_t max_level)
    {
        const std::size_t hi = std::max(max_level, lca.level());
        pyramid.clear();
        pyramid.set_origin_point(lca.origin_point());
        pyramid.set_scaling_factor(lca.scaling_factor());

        for (std::size_t level = 0; level <= hi; ++level)
        {
            if (level == lca.level())
            {
                pyramid[level] = lca;
                continue;
            }

            self(lca).on(level)(
                [&](const auto& i, const auto& index)
                {
                    pyramid[level].add_interval_back(i, index);
                });
        }
    }

    template <class CellArray, class LCA>
    CellArray make_level_pyramid(const LCA& lca, std::size_t max_level)
    {
        CellArray pyramid;
        build_level_pyramid(pyramid, lca, max_level);
        return pyramid;
    }
}
