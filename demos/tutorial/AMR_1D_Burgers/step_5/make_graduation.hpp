// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

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
                auto mask = (tag(level, i) & static_cast<int>(samurai::CellFlag::refine));

                for (int ii = -1; ii <= 1; ++ii)
                {
                    samurai::apply_on_masked(tag(level, i + ii),
                                             mask,
                                             [](auto& e)
                                             {
                                                 e |= static_cast<int>(samurai::CellFlag::keep);
                                             });
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
                auto mask = (tag(level, 2 * i) & static_cast<int>(samurai::CellFlag::keep))
                          | (tag(level, 2 * i + 1) & static_cast<int>(samurai::CellFlag::keep));

                samurai::apply_on_masked(tag(level, 2 * i),
                                         mask,
                                         [](auto& e)
                                         {
                                             e |= static_cast<int>(samurai::CellFlag::keep);
                                         });
                samurai::apply_on_masked(tag(level, 2 * i + 1),
                                         mask,
                                         [](auto& e)
                                         {
                                             e |= static_cast<int>(samurai::CellFlag::keep);
                                         });
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
                    auto mask_refine = tag(level, interval - s) & static_cast<int>(samurai::CellFlag::refine);
                    auto half_i      = interval >> 1;
                    samurai::apply_on_masked(tag(level - 1, half_i),
                                             mask_refine,
                                             [](auto& e)
                                             {
                                                 e |= static_cast<int>(samurai::CellFlag::refine);
                                             });

                    auto mask_keep = tag(level, interval - s) & static_cast<int>(samurai::CellFlag::keep);
                    samurai::apply_on_masked(tag(level - 1, half_i),
                                             mask_keep,
                                             [](auto& e)
                                             {
                                                 e |= static_cast<int>(samurai::CellFlag::keep);
                                             });
                });
        }
    }
}
