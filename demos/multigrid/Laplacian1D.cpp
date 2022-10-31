#pragma once
#include "samurai_new/multigrid/petsc/utils.hpp"
#include "utils.hpp"
#include "samurai_new/boundary.hpp"
#include "samurai_new/indices.hpp"

//-------------------//
//     Laplacian     //
// Implementation 1D //
//-------------------//

// Prediction (order 1)
template<class Mesh>
void assemble_prediction_impl(std::integral_constant<std::size_t, 1>, Mat& A, Mesh& mesh)
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
        set([&](const auto& i, const auto&)
        {
            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = static_cast<int>(mesh.get_index(level, ii));
                MatSetValue(A, i_cell, i_cell, 1., INSERT_VALUES);

                int sign_i = (ii & 1)? -1: 1;

                for(int is = -1; is<2; ++is)
                {
                    auto i1 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) + is));
                    double v = -sign_i*pred[is + 1];
                    MatSetValue(A, i_cell, i1, v, INSERT_VALUES);
                }

                auto i0 = static_cast<int>(mesh.get_index(level - 1, (ii>>1)));
                MatSetValue(A, i_cell, i0, -1., INSERT_VALUES);
            }
        });
    }
}

// Boundary

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
void assemble_boundary_condition_impl(std::integral_constant<std::size_t, 1>, Mat& A, Mesh& mesh, DirichletEnforcement dirichlet_enfcmt)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    static constexpr std::size_t dim = Mesh::dim;
    static constexpr PetscInt scheme_stencil_size = 1 + 2*dim;

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
            //                                                              // out_cell, in_cell
            //     samurai_new::StencilIndices<PetscInt, dim, 2> out_in_stencil({{0}, {-out_vect[0]}});
            //     samurai_new::for_each_stencil<PetscInt>(mesh, level, i, index, out_in_stencil, [&] (const std::array<PetscInt, 2>& indices)
            //     {
            //         auto& out_cell = indices[0];
            //         auto& in_cell  = indices[1];
            //         MatSetValue(A, out_cell, out_cell,                               1, INSERT_VALUES); // The outer unknown is eliminated from the system
            //         MatSetValue(A,  in_cell, in_cell , scheme_stencil_size*one_over_h2, INSERT_VALUES);
            //         MatSetValue(A,  in_cell, out_cell,                               0, INSERT_VALUES); // Remove the coefficient that was added before
            //     });
            // }
                                                                     // out_cell, in_cell
            samurai_new::StencilIndices<PetscInt, dim, 2> out_in_stencil({{0}, {-out_vect[0]}});
            auto n_zeros = samurai_new::number_of_zeros(out_vect);
            double in_diag_value = (scheme_stencil_size-1) + dim - n_zeros;
            in_diag_value *= one_over_h2;
            samurai_new::for_each_stencil<PetscInt>(mesh, level, i, index, out_in_stencil, [&] (const std::array<PetscInt, 2>& indices)
            {
                auto& out_cell = indices[0];
                auto& in_cell  = indices[1];
                MatSetValue(A, out_cell, out_cell,             1, INSERT_VALUES); // The outer unknown is eliminated from the system
                MatSetValue(A,  in_cell, in_cell , in_diag_value, INSERT_VALUES);
                MatSetValue(A,  in_cell, out_cell,             0, INSERT_VALUES); // Remove the coefficient that was added before
            });
        });
    });

    MatSetOption(A, MAT_SPD, PETSC_TRUE);
}*/


template<class Field>
Vec assemble_rhs_impl(std::integral_constant<std::size_t, 1>, Field& rhs_field, DirichletEnforcement /*dirichlet_enfcmt*/)
{
    using Mesh = typename Field::mesh_t;
    using mesh_id_t = typename Mesh::mesh_id_t;

    Mesh& mesh = rhs_field.mesh();

    for(std::size_t level=mesh.min_level(); level<=mesh.max_level(); ++level)
    {
        auto set = samurai::difference(mesh[mesh_id_t::reference][level],
                                    mesh[mesh_id_t::cells][level]);
        set([&](const auto& i, const auto&)
        {
            rhs_field(level, i) = 0.;
        });
    }

    return samurai_new::petsc::create_petsc_vector_from(rhs_field);
}