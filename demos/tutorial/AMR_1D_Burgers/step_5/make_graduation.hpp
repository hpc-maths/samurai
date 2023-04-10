// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>

#include <xtensor/xmasked_view.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_flag.hpp>
#include <samurai/subset/subset_op.hpp>

template <class Field>
void make_graduation(Field& tag)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    auto& mesh = tag.mesh();
    for (std::size_t level = mesh.max_level(); level >= 1; --level)
    {
        /**
         *
         *        |-----|-----|                                  |-----|-----|
         *                                    --------------->
         *                                                             K
         *        |===========|-----------| |===========|-----------|
         */

        auto ghost_subset = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);

        ghost_subset(
            [&](const auto& i, const auto&)
            {
                tag(level - 1, i) |= static_cast<int>(samurai::CellFlag::keep);
            });

        /**
         *                 R                                 K     R     K
         *        |-----|-----|=====|   --------------->  |-----|-----|=====|
         *
         */
        auto leaves = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        leaves(
            [&](const auto& i, const auto&)
            {
                xt::xtensor<bool, 1> mask = (tag(level, i) & static_cast<int>(samurai::CellFlag::refine));

                for (int ii = -1; ii <= 1; ++ii)
                {
                    xt::masked_view(tag(level, i + ii), mask) |= static_cast<int>(samurai::CellFlag::keep);
                }
            });

        /**
         *      K     C                          K     K
         *   |-----|-----|   -------------->  |-----|-----|
         *
         *   |-----------|
         *
         */
        leaves.on(level - 1)(
            [&](const auto& i, const auto&)
            {
                xt::xtensor<bool, 1> mask = (tag(level, 2 * i) & static_cast<int>(samurai::CellFlag::keep))
                                          | (tag(level, 2 * i + 1) & static_cast<int>(samurai::CellFlag::keep));

                xt::masked_view(tag(level, 2 * i), mask) |= static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, 2 * i + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
            });

        const std::array<int, 2> stencil{1, -1};

        /**
         * Case 1
         * ======
         *                   R     K R     K
         *                |-----|-----|   --------------> |-----|-----| C or K R
         *   |-----------|                                        |-----------|
         *
         * Case 2
         * ======
         *                   K     K K     K
         *                |-----|-----|   --------------> |-----|-----| C K
         *   |-----------|                                        |-----------|
         *
         */

        for (auto s : stencil)
        {
            auto subset = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::cells][level - 1]).on(level);

            subset(
                [&](const auto& interval, const auto&)
                {
                    auto mask   = tag(level, interval - s) & static_cast<int>(samurai::CellFlag::refine);
                    auto half_i = interval >> 1;
                    xt::masked_view(tag(level - 1, half_i), mask) |= static_cast<int>(samurai::CellFlag::refine);

                    mask = tag(level, interval - s) & static_cast<int>(samurai::CellFlag::keep);
                    xt::masked_view(tag(level - 1, half_i), mask) |= static_cast<int>(samurai::CellFlag::keep);
                });
        }
    }
}
