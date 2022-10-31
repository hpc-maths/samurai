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

// Dirichlet condition enforcement.
// Penalty method: 
//         The diagonal coefficient of the Dirichlet rows is replaced with itself plus a penalty.
//         The other coeff in the row is unchanged.
// Non-symmetric: 
//         The diagonal coefficient of the Dirichlet rows is replaced with 1. The other coeff in the row is set to 0.
//         This method kills the symmetry of the matrix, but used in an iterative solver it's fine because the residual is always 0
//         on the Dirichlet unknowns, so the behaviour of the solver is the same as if the unknowns had been eliminated.
//         (cf. Ern-Guermont 2004 - Theory and practice of FE, §8.4.3 p. 378)
/*template<class Mesh>
void assemble_boundary_condition_impl(std::integral_constant<std::size_t, 2>, Mat& A, Mesh& mesh, DirichletEnforcement dirichlet_enfcmt)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    static constexpr std::size_t dim = Mesh::dim;
    static constexpr PetscInt scheme_stencil_size = 1 + 2*dim;
    static constexpr double penalty_coeff = 1000;

    samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
    {
        double one_over_h2 = 1/(h*h);

        samurai_new::out_boundary(mesh, level, 
        [&] (const auto& i, const auto& index, const auto& out_vect)
        {
            // if (!samurai_new::is_cartesian_direction(out_vect)) // corners
            // {
            //     samurai_new::for_each_cell<PetscInt>(mesh, level, i, index, 
            //     [&] (PetscInt out_cell)
            //     {
            //         MatSetValue(A, out_cell, out_cell, 1, INSERT_VALUES);
            //     });
            // }
            // else // Cartesian direction
            // {
            //     bool neumann = false;
            //     if (neumann || dirichlet_enfcmt == Penalization)
            //     {
            //         double one_over_h2 = 1/(h*h);
            //         double v_diag = one_over_h2;
            //         if (dirichlet_enfcmt == Penalization)
            //             v_diag *= (1 + penalty_coeff);
            //                                                                  // out_cell,           in_cell
            //         samurai_new::StencilIndices<PetscInt, dim, 2> out_in_stencil({{0, 0}, {-out_vect[0], -out_vect[1]}});
            //         samurai_new::for_each_stencil<PetscInt>(mesh, level, i, index, out_in_stencil, [&] (const std::array<PetscInt, 2>& indices)
            //         {
            //             auto& out_cell = indices[0];
            //             auto& in_cell  = indices[1];
            //             MatSetValue(A, out_cell, out_cell,       v_diag, INSERT_VALUES);
            //             MatSetValue(A, out_cell, in_cell , -one_over_h2, INSERT_VALUES);
            //         });
            //     }
            //     else if (dirichlet_enfcmt == Elimination)
            //     {
            //                                                                  // out_cell,           in_cell
            //         samurai_new::StencilIndices<PetscInt, dim, 2> out_in_stencil({{0, 0}, {-out_vect[0], -out_vect[1]}});
            //         samurai_new::for_each_stencil<PetscInt>(mesh, level, i, index, out_in_stencil, [&] (const std::array<PetscInt, 2>& indices)
            //         {
            //             auto& out_cell = indices[0];
            //             auto& in_cell  = indices[1];
            //             MatSetValue(A, out_cell, out_cell, 1, INSERT_VALUES);
            //             MatSetValue(A, out_cell, in_cell , 0, INSERT_VALUES); // Remove the coefficient that was added before
            //         });
            //     }
            //     else if (dirichlet_enfcmt == OnesOnDiagonal)
            //     {
            //         samurai_new::for_each_cell<PetscInt>(mesh, level, i, index, 
            //         [&] (PetscInt out_cell)
            //         {
            //             MatSetValue(A, out_cell, out_cell, 1, INSERT_VALUES);
            //         });
            //     }
            // }
                                                                             // out_cell,           in_cell
            samurai_new::StencilIndices<PetscInt, dim, 2> out_in_stencil({{0, 0}, {-out_vect[0], -out_vect[1]}});
            auto n_zeros = samurai_new::number_of_zeros(out_vect);
            double in_diag_value = (scheme_stencil_size-1) + dim - n_zeros;
            in_diag_value *= one_over_h2;
            samurai_new::for_each_stencil<PetscInt>(mesh, level, i, index, out_in_stencil, [&] (const std::array<PetscInt, 2>& indices)
            {
                auto& out_cell = indices[0];
                auto& in_cell  = indices[1];
                MatSetValue(A, out_cell, out_cell,             1, INSERT_VALUES); // The outer unknown is eliminated from the system
                MatSetValue(A,  in_cell, in_cell , in_diag_value, INSERT_VALUES);
                if (n_zeros > 0) // if n_zeros==0, then its a corner: out_cell is not in the stencil of in_cell, so nothing to remove.
                {  
                    MatSetValue(A,  in_cell, out_cell,         0, INSERT_VALUES); // Remove the coefficient that was added before
                }
            });
        });
    });

    if (dirichlet_enfcmt != OnesOnDiagonal)
        MatSetOption(A, MAT_SPD, PETSC_TRUE);
}*/


template<class Field>
Vec assemble_rhs_impl(std::integral_constant<std::size_t, 2>, Field& rhs_field, DirichletEnforcement /*dirichlet_enfcmt*/)
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
            rhs_field(level, i, j) = 0.;
        });
    }

    return samurai_new::petsc::create_petsc_vector_from(rhs_field);
}