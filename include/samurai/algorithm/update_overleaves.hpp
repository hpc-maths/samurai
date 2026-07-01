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
    template <class Field>
    void update_overleaves_mr(Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh            = field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        update_outer_ghosts(field);
        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            // These are the overleaves which are nothing else
            // because when this procedure is called all the rest
            // should be already with the right value.
            auto overleaves_to_predict = difference(difference(mesh[mesh_id_t::overleaves][level], mesh[mesh_id_t::cells_and_ghosts][level]),
                                                    mesh[mesh_id_t::proj_cells][level]);

            overleaves_to_predict.apply_op(prediction<1, false>(field));
        }
    }
}
