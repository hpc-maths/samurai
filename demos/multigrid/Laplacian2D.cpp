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
void assemble_prediction_impl(std::integral_constant<std::size_t, 2>, Mat& A, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();
    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {
        auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                         mesh[mesh_id_t::cells][level-1])
                .on(level);

        std::array<double, 3> pred{{1./8, 0, -1./8}};
        set([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            int sign_j = (j & 1)? -1: 1;

            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = static_cast<PetscInt>(mesh.get_index(level, ii, j));
                MatSetValue(A, i_cell, i_cell, 1, INSERT_VALUES);

                int sign_i = (ii & 1)? -1: 1;

                for(int is = -1; is<2; ++is)
                {
                    auto i1 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1), (j>>1) + is));
                    MatSetValue(A, i_cell, i1, -sign_j*pred[is + 1], INSERT_VALUES);

                    i1 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) + is, (j>>1)));
                    MatSetValue(A, i_cell, i1, -sign_i*pred[is + 1], INSERT_VALUES);
                }

                auto i1 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) - 1));
                auto i2 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) - 1));
                auto i3 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) + 1));
                auto i4 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) + 1));

                MatSetValue(A, i_cell, i1, sign_i*sign_j*pred[0]*pred[0], INSERT_VALUES);
                MatSetValue(A, i_cell, i2, sign_i*sign_j*pred[2]*pred[0], INSERT_VALUES);
                MatSetValue(A, i_cell, i3, sign_i*sign_j*pred[0]*pred[2], INSERT_VALUES);
                MatSetValue(A, i_cell, i4, sign_i*sign_j*pred[2]*pred[2], INSERT_VALUES);

                auto i0 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1), (j>>1)));
                MatSetValue(A, i_cell, i0, -1, INSERT_VALUES);
            }
        });
    }
}