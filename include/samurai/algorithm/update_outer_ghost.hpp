// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../algorithm.hpp"
#include "../bc/apply_field_bc.hpp"

namespace samurai
{
    /**
     * Fill the outer ghosts of one level: the cells outside the domain that the mesh holds next
     * to the cells of that level inside it.
     *
     * This is the **physical extension** of the field - the values the user's boundary
     * conditions prescribe - and it is evaluated at every level independently, from the cells
     * of that level: the boundary condition first, then the polynomial extrapolation of the
     * layers it leaves unfilled, then the diagonal directions. Nothing here transfers a value
     * from one level to another: a coarse level under a refined boundary gets its outer ghosts
     * from the boundary condition applied to its own projected values, a fine layer under a
     * coarse cell from the condition applied to its own predicted values. It is therefore
     * called once per level, when every cell of the level inside the domain has its value:
     * after the prediction ghosts of the level are filled, in the bottom-up pass of the ghost
     * update.
     *
     * What used to be here transferred them: the outer ghosts of a coarse level were averaged
     * from the finer level's (`project_bc`), the fine ones under a coarse cell copied from it
     * (`predict_bc`, at order 0), the corner ghosts projected two levels down with a hardcoded
     * 2 - an orchestration from the fine levels to the coarse ones, with per-cell existence
     * guards and branches on an `is_box()` flag that is not a computed property. All of it
     * existed for one reader, the prediction stencil, which needed the outer ghosts of a coarse
     * level to hold what the fine ones did. Prediction now shifts its stencil inward near a
     * boundary and reads only cells the domain holds wherever the domain is wide enough for
     * that (prediction_shifts.hpp); where it is not - a level only `2r` cells wide or less,
     * which is the coarsest level of a small box - the stencil stays centred and reads what is
     * written here, at that level, by that level's own boundary condition.
     *
     * A condition is applied around the cells whose whole stencil the mesh holds at this level
     * (cells_holding_stencil): a projection ghost next to an obstacle does not always have, at
     * its own level, the inner neighbours a high-order condition reads.
     */
    template <class Field>
    void update_outer_ghosts(std::size_t level, Field& field)
    {
        constexpr std::size_t dim = Field::dim;

        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        // A level the mesh does not hold at all has no outer ghost to fill; the ghost update
        // visits every level from 0 up, and a mesh refined everywhere holds only a few of them.
        if (level > mesh.max_level() || mesh[mesh_id_t::reference][level].empty())
        {
            return;
        }

        for_each_cartesian_direction<dim>(
            [&](auto direction_index, const auto& direction)
            {
                if (!mesh.is_periodic(direction_index))
                {
                    apply_field_bc(level, direction, field);
                    // The layers beyond those the boundary condition fills, up to the ghost
                    // width the scheme needs.
                    update_further_ghosts_by_polynomial_extrapolation(level, direction, field);
                }
            });

        if constexpr (dim > 1)
        {
            for_each_diagonal_direction<dim>(
                [&](auto& direction)
                {
                    // A periodic direction has no real boundary, so no corner ghost to fill.
                    bool any_periodic = false;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        if (direction[d] != 0 && mesh.is_periodic(d))
                        {
                            any_periodic = true;
                            break;
                        }
                    }
                    if (!any_periodic)
                    {
                        // No finite-volume scheme reads a diagonal ghost, so this is for the
                        // schemes that do: a lattice-Boltzmann stream with diagonal velocities.
                        // The extrapolation is the fallback, overwritten by a boundary condition
                        // that owns the diagonal directions (see Bc::fills_diagonal_directions).
                        update_outer_corners_by_polynomial_extrapolation(level, direction, field);
                        apply_field_bc_diagonal(level, direction, field);
                    }
                });
        }
    }

    /**
     * Fill the outer ghosts of every level: for a field whose inner ghosts are all in place.
     * The levels are independent, so the order does not matter.
     */
    template <class Field>
    void update_outer_ghosts(Field& field)
    {
        auto& mesh = field.mesh();

        for (std::size_t level = 0; level <= mesh.max_level(); ++level)
        {
            update_outer_ghosts(level, field);
        }
    }

    template <class Field, class... Fields>
    void update_outer_ghosts(Field& field, Fields&... fields)
    {
        update_outer_ghosts(field);
        update_outer_ghosts(fields...);
    }

    template <class Field, class... Fields>
    void update_outer_ghosts(std::size_t level, Field& field, Fields&... fields)
    {
        update_outer_ghosts(level, field);
        update_outer_ghosts(level, fields...);
    }
}
