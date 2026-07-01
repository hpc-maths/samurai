// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>

#include <xtensor/containers/xfixed.hpp>

#include "../algorithm.hpp"
#include "../bc/apply_field_bc.hpp"
#include "../concepts.hpp"
#include "../field.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"
#include "../subset/node.hpp"
#include "../timers.hpp"
#include "graduation.hpp"
#include "utils.hpp"

#ifndef NDEBUG
#include "../io/hdf5.hpp"
#endif

using namespace xt::placeholders;

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <xtensor/views/xmasked_view.hpp>
namespace mpi = boost::mpi;
#endif

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

    template <class Field>
    void update_ghost_mro(Field& field)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config_t::prediction_stencil_radius;
        auto& mesh                       = field.mesh();

        std::size_t max_level = mesh.max_level();

        update_outer_ghosts(max_level, field);
        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(projection(field));
            update_outer_ghosts(level - 1, field);
        }

        update_outer_ghosts(0, field);
        for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= max_level; ++level)
        {
            // We eliminate the overleaves from the computation since they
            // are done separately
            // auto expr =
            // difference(intersection(difference(mesh[mesh_id_t::all_cells][level],
            //                                                union_(mesh[mesh_id_t::cells][level],
            //                                                       mesh[mesh_id_t::proj_cells][level])),
            //                                     mesh.domain()),
            //                        difference(mesh[mesh_id_t::overleaves][level],
            //                                   union_(mesh[mesh_id_t::union_cells][level],
            //                                          mesh[mesh_id_t::cells_and_ghosts][level])))
            //             .on(level);

            auto expr = intersection(
                difference(mesh[mesh_id_t::all_cells][level], union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level])),
                self(mesh.domain()).on(level));

            expr.apply_op(prediction<pred_order, false>(field));
        }
    }
}
