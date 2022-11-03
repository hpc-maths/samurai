#pragma once
#include "samurai_new/multigrid/petsc/utils.hpp"
#include "utils.hpp"
#include "samurai_new/boundary.hpp"
#include "samurai_new/indices.hpp"

//-------------------//
//     Laplacian     //
// Implementation 3D //
//-------------------//

// Prediction (order 1)
template<class Mesh>
void assemble_prediction_impl(std::integral_constant<std::size_t, 3>, Mat& /*A*/, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();
    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {
        auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                         mesh[mesh_id_t::cells][level-1])
                .on(level);

        //std::array<double, 3> pred{{1./8, 0, -1./8}};
        set([&](const auto&, const auto&)
        {
            assert(false && "non implemented");
        });
    }
}