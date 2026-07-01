// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../algorithm.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"

namespace samurai
{
    template <class Field, class... Fields>
    void update_ghost(Field& field, Fields&... fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config_t::prediction_stencil_radius;

        auto& mesh            = field.mesh();
        std::size_t max_level = mesh.max_level();

        update_outer_ghosts(max_level, field, fields...);
        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::proj_cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(variadic_projection(field, fields...));
            update_outer_ghosts(level - 1, field, fields...);
        }

        update_outer_ghosts(0, field, fields...);
        for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= max_level; ++level)
        {
            auto set_at_level = intersection(mesh[mesh_id_t::pred_cells][level], mesh[mesh_id_t::reference][level - 1]).on(level);
            set_at_level.apply_op(variadic_prediction<pred_order, false>(field, fields...));
        }

        field.ghosts_updated() = true;
        ((fields.ghosts_updated() = true), ...);
    }
}
