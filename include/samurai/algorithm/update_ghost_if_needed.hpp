// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <type_traits>

#include "../amr/mesh.hpp"
#include "update_basic_ghost.hpp"
#include "update_ghost_mr.hpp"

namespace samurai
{
    /**
     * Refresh the ghosts of the fields whose ghosts are out of date, with the ghost
     * update that matches their mesh: @c update_ghost for an AMR mesh, whose projection
     * cells follow another convention than the multiresolution mesh, @c update_ghost_mr
     * otherwise. Operators that accept any mesh call this function.
     */
    template <class Field>
    void update_ghost_if_needed(Field& field)
    {
        if (field.ghosts_updated())
        {
            return;
        }
        if constexpr (std::is_same_v<typename Field::mesh_t::mesh_id_t, amr::AMR_Id>)
        {
            update_ghost(field);
        }
        else
        {
            update_ghost_mr(field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_if_needed(Field& field, Fields&... other_fields)
    {
        update_ghost_if_needed(field);
        update_ghost_if_needed(other_fields...);
    }
}
