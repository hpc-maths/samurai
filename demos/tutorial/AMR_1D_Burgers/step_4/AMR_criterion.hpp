// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/algorithm.hpp>
#include <samurai/cell_flag.hpp>

/**
 * AMR criterion
 *
 * Split the cell (level, i) if |\partial_x f(level, i)| > \delta
 *
 * with \delta = 0.01
 *
 * and \partial_x f(level, i) = (f(level, i+1) - f(level, i-1))/(2 \Delta x)
 *
 * \Delta x = 2^{-level}
 *
 */

template <class Field, class Tag>
void AMR_criterion(const Field& f, Tag& tag)
{
    using namespace samurai::math;
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    auto& mesh = f.mesh();

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    tag.fill(static_cast<int>(samurai::CellFlag::keep));

    samurai::for_each_interval(mesh[mesh_id_t::cells],
                               [&](std::size_t level, auto& i, auto&)
                               {
                                   const double dx = mesh.cell_length(level);

                                   auto der_approx = samurai::eval(abs((f(level, i + 1) - f(level, i - 1)) / (2. * dx)));
                                   auto mask       = der_approx > 0.01;

                                   if (level < max_level)
                                   {
                                       samurai::apply_on_masked(tag(level, i),
                                                                mask,
                                                                [](auto& e)
                                                                {
                                                                    e = static_cast<int>(samurai::CellFlag::refine);
                                                                });
                                   }
                                   if (level > min_level)
                                   {
                                       samurai::apply_on_masked(tag(level, i),
                                                                !mask,
                                                                [](auto& e)
                                                                {
                                                                    e = static_cast<int>(samurai::CellFlag::coarsen);
                                                                });
                                   }
                               });
}
