#pragma once

#include <spdlog/spdlog.h>

#include "../field.hpp"

namespace samurai
{
    template<class Field>
    inline void mr_projection(Field &field)
    {
        spdlog::debug("Make projection");

        auto mesh = field.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= min_level; --level)
        {
            auto expr = intersection(mesh[mesh_id_t::all_cells][level],
                                     mesh[mesh_id_t::proj_cells][level - 1])
                       .on(level - 1);

            expr.apply_op(projection(field));
        }
    }

    template<class Field, class Func>
    inline void mr_prediction(Field &field, Func&& update_bc_for_level)
    {
        spdlog::debug("Make prediction");

        auto mesh = field.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            // We eliminate the overleaves from the computation since they
            // are done separately
            auto expr = difference(intersection(difference(mesh[mesh_id_t::all_cells][level],
                                                           union_(mesh[mesh_id_t::cells][level],
                                                                  mesh[mesh_id_t::proj_cells][level])),
                                                mesh.domain()),
                                   difference(mesh[mesh_id_t::overleaves][level],
                                              union_(mesh[mesh_id_t::union_cells][level],
                                                     mesh[mesh_id_t::cells_and_ghosts][level])))
                        .on(level);

            expr.apply_op(prediction(field));
            update_bc_for_level(field, level);
        }
    }

    template<class Field, class Func>
    inline void mr_prediction_overleaves(Field &field, Func&& update_bc_for_level)
    {
        spdlog::debug("Make prediction on the overleaves which are not already available");

        auto mesh = field.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            // These are the overleaves which are nothing else
            // because when this procedure is called all the rest
            // should be already with the right value.
            auto overleaves_to_predict = difference(difference(mesh[mesh_id_t::overleaves][level],
                                                               mesh[mesh_id_t::cells_and_ghosts][level]),
                                                    mesh[mesh_id_t::proj_cells][level]);

            overleaves_to_predict.apply_op(prediction(field));
            update_bc_for_level(field, level);
        }
    }
} // namespace samurai
