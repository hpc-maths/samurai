#pragma once
#include "samurai_new/multigrid/petsc/utils.hpp"
#include "utils.hpp"
#include "samurai_new/boundary.hpp"
#include "samurai_new/indices.hpp"

//-------------------//
//     Laplacian     //
// Implementation 2D //
//-------------------//

template<class Mesh>
std::vector<int> preallocate_matrix_impl(std::integral_constant<std::size_t, 2>, Mesh& mesh, DirichletEnforcement dirichlet_enfcmt)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    std::size_t n = mesh.nb_cells();

    static constexpr PetscInt scheme_stencil_size =  5; // 5-point stencil Finite Volume scheme
    static constexpr PetscInt proj_stencil_size   =  5; // cell + 4 children
    static constexpr PetscInt pred_stencil_size   = 10; // cell + parent's 9-point stencil (prediction order 1)

    std::vector<PetscInt> nnz(n, 1);

    // Cells on the same level
    samurai_new::for_each_cell<std::size_t>(mesh, mesh[mesh_id_t::cells], [&](std::size_t cell)
    {
        nnz[cell] = scheme_stencil_size;
    });

    // Projection
    samurai_new::for_each_cell_having_children<std::size_t>(mesh, [&] (std::size_t cell)
    {
        nnz[cell] = proj_stencil_size;
    });

    // Prediction
    samurai_new::for_each_cell_having_parent<std::size_t>(mesh, [&] (std::size_t cell)
    {
        nnz[cell] = pred_stencil_size;
    });

    // Boundary conditions
    if (dirichlet_enfcmt != OnesOnDiagonal)
    {
        samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double)
        {
            samurai_new::out_boundary(mesh, level, 
            [&] (const auto& i, const auto& index, const auto& out_vect)
            {
                auto j = index[0];
                if (out_vect[0] == 0 || out_vect[1] == 0) // Cartesian direction
                {
                    samurai_new::for_each_cell<std::size_t>(mesh, level, i, j, 
                    [&] (std::size_t i_out)
                    {
                        nnz[i_out] = 2;
                    });
                }
            });
        });
    }

    return nnz;
}




template<class Mesh>
PetscErrorCode assemble_matrix_impl(std::integral_constant<std::size_t, 2>, Mat& A, Mesh& mesh, DirichletEnforcement dirichlet_enfcmt)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    static constexpr std::size_t dim = Mesh::dim;

    // For each group of cells given by stencil_shape, we want to capture the indices as PetscInt.
    // 5-point stencil:                                    center,  bottom, right,   top,     left
    samurai_new::StencilIndices<PetscInt, dim, 5> stencil({{0, 0}, {0, -1}, {1, 0}, {0, 1}, {-1, 0}});

    samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
    {
        double one_over_h2 = 1/(h*h);

        double coeffs_stencil_row[3];
        coeffs_stencil_row[0] =    -one_over_h2;
        coeffs_stencil_row[1] = 4 * one_over_h2;
        coeffs_stencil_row[2] =    -one_over_h2;

        samurai_new::for_each_stencil<PetscInt>(mesh, mesh[mesh_id_t::cells], level, stencil,
        [&] (const std::array<PetscInt, 5>& indices)
        {
            auto center = indices[0];
            auto bottom = indices[1];
            auto right  = indices[2];
            auto top    = indices[3];
            auto left   = indices[4];

            PetscInt stencil_row[3];
            stencil_row[0] = left;
            stencil_row[1] = center;
            stencil_row[2] = right;
            MatSetValues(A, 1, &center, 3, stencil_row, coeffs_stencil_row, INSERT_VALUES);
            MatSetValue(A, center, top   , -one_over_h2, INSERT_VALUES);
            MatSetValue(A, center, bottom, -one_over_h2, INSERT_VALUES);
        });
    });

    // Projection
    samurai_new::for_each_cell_and_children<PetscInt>(mesh, 
    [&] (PetscInt cell, const std::array<PetscInt, 4>& children)
    {
        MatSetValue(A, cell, cell, 1, INSERT_VALUES);
        for (unsigned int i=0; i<4; ++i)
        {
            MatSetValue(A, cell, children[i], -0.25, INSERT_VALUES);
        }
    });

    // Prediction (order 1)
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
    static constexpr double penalty_coeff = 1000;
    samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
    {
        samurai_new::out_boundary(mesh, level, 
        [&] (const auto& i, const auto& index, const auto& out_vect)
        {
            auto j = index[0];

            if (out_vect[0] != 0 && out_vect[1] != 0) // corners
            {
                samurai_new::for_each_cell<PetscInt>(mesh, level, i, j, 
                [&] (PetscInt out_cell)
                {
                    MatSetValue(A, out_cell, out_cell, 1, INSERT_VALUES);
                });
            }
            else // Cartesian direction
            {
                bool neumann = false;
                if (neumann || dirichlet_enfcmt == Penalization)
                {
                    double one_over_h2 = 1/(h*h);
                    double v_diag = one_over_h2;
                    if (dirichlet_enfcmt == Penalization)
                        v_diag *= (1 + penalty_coeff);
                                                                             // out_cell,           in_cell
                    samurai_new::StencilIndices<PetscInt, dim, 2> out_in_stencil({{0, 0}, {-out_vect[0], -out_vect[1]}});
                    samurai_new::for_each_stencil<PetscInt>(mesh, level, i, index, out_in_stencil, [&] (const std::array<PetscInt, 2>& indices)
                    {
                        auto& out_cell = indices[0];
                        auto& in_cell  = indices[1];
                        MatSetValue(A, out_cell, out_cell,       v_diag, INSERT_VALUES);
                        MatSetValue(A, out_cell, in_cell , -one_over_h2, INSERT_VALUES);
                    });
                }
                else if (dirichlet_enfcmt == Elimination)
                {
                                                                             // out_cell,           in_cell
                    samurai_new::StencilIndices<PetscInt, dim, 2> out_in_stencil({{0, 0}, {-out_vect[0], -out_vect[1]}});
                    samurai_new::for_each_stencil<PetscInt>(mesh, level, i, index, out_in_stencil, [&] (const std::array<PetscInt, 2>& indices)
                    {
                        auto& out_cell = indices[0];
                        auto& in_cell  = indices[1];
                        MatSetValue(A, out_cell, out_cell, 1, INSERT_VALUES);
                        MatSetValue(A, out_cell, in_cell , 0, INSERT_VALUES); // Remove the coefficient that was added before
                    });
                }
                else if (dirichlet_enfcmt == OnesOnDiagonal)
                {
                    samurai_new::for_each_cell<PetscInt>(mesh, level, i, j, 
                    [&] (PetscInt out_cell)
                    {
                        MatSetValue(A, out_cell, out_cell, 1, INSERT_VALUES);
                    });
                }
            }
        });
    });


    if (dirichlet_enfcmt != OnesOnDiagonal)
        MatSetOption(A, MAT_SPD, PETSC_TRUE);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    PetscFunctionReturn(0);
}


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