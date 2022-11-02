#pragma once
#include "samurai_new/multigrid/petsc/utils.hpp"
#include "utils.hpp"
#include "samurai_new/boundary.hpp"
#include "samurai_new/indices.hpp"

//-------------------//
//     Laplacian     //
// Implementation 2D //
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

template<class Field>
Vec assemble_rhs_impl(std::integral_constant<std::size_t, 3>, Field& rhs_field, DirichletEnforcement /*dirichlet_enfcmt*/)
{
    using Mesh = typename Field::mesh_t;
    using mesh_id_t = typename Mesh::mesh_id_t;

    Mesh& mesh = rhs_field.mesh();

    for(std::size_t level=mesh.min_level(); level<=mesh.max_level(); ++level)
    {
        auto set = samurai::difference(mesh[mesh_id_t::reference][level],
                                    mesh[mesh_id_t::cells][level]);
        set([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            auto k = index[1];
            rhs_field(level, i, j, k) = 0.;
        });
    }

    return samurai_new::petsc::create_petsc_vector_from(rhs_field);
}