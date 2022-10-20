#pragma once
#include "samurai_new/multigrid/petsc/utils.hpp"
#include "utils.hpp"
#include "samurai_new/boundary.hpp"

//-------------------//
//     Laplacian     //
// Implementation 2D //
//-------------------//

template<class Mesh>
std::vector<int> preallocate_matrix_impl(std::integral_constant<std::size_t, 2>, Mesh& mesh, DirichletEnforcement /*dirichlet_enfcmt*/)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    std::size_t n = mesh.nb_cells();

    std::vector<int> nnz(n, 1);

    samurai::for_each_interval(mesh[mesh_id_t::cells_and_ghosts], [&](std::size_t level, const auto& i, const auto& index)
    {
        auto j = index[0];
        for(int ii=i.start; ii<i.end; ++ii)
        {
            auto i_j = mesh.get_index(level, ii, j);
            nnz[i_j] = 5;
        }
    });

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();

    for(std::size_t level=min_level; level<max_level; ++level)
    {
        auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                         mesh[mesh_id_t::cells][level+1])
                    .on(level);

        set([&](const auto& i, const auto& index)
        {
            auto j = index[0];

            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = mesh.get_index(level, ii, j);
                nnz[i_cell] = 5;
            }
        });
    }

    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {
        auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                         mesh[mesh_id_t::cells][level-1])
                .on(level);

        set([&](const auto& i, const auto& index)
        {
            auto j = index[0];

            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = mesh.get_index(level, ii, j);
                nnz[i_cell] = 10;
            }
        });
    }

    
    for(std::size_t level=min_level; level<=max_level; ++level)
    {
        auto set = samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level],
                                    mesh.domain())
                .on(level);

        set([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = mesh.get_index(level, ii, j);
                nnz[i_cell] = 5;
            }
        });
    }

    return nnz;
}



inline double off_diag_coeff_2D(std::size_t level)
{
    //double dx = 1./(1<<level);
    //double one_over_dx2 = 1./(dx*dx);
    std::size_t one_over_dx = 1<<level;
    return -static_cast<double>(one_over_dx*one_over_dx);

    //double v_diag = 4*one_over_dx2;
    //double v_off = -one_over_dx2;
}



template<class Mesh>
PetscErrorCode assemble_matrix_impl(std::integral_constant<std::size_t, 2>, Mat& A, Mesh& mesh, DirichletEnforcement dirichlet_enfcmt)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    //using interval_t = typename Mesh::interval_t;

    constexpr int stencil_size = 4;

    auto n = static_cast<PetscInt>(mesh.nb_cells());

    for(int i=0; i<n; ++i)
    {
        MatSetValue(A, i, i, 1, INSERT_VALUES);
    }

    auto& cells = dirichlet_enfcmt == Penalization ? mesh[mesh_id_t::cells] : mesh[mesh_id_t::cells];

    samurai::for_each_interval(cells, [&](std::size_t level, const auto& i, const auto& index)
    {
        auto j = index[0];

        double v_off = off_diag_coeff_2D(level);
        double v_diag = -stencil_size*v_off;

        for(int ii=i.start; ii<i.end; ++ii)
        {
            auto i_j = static_cast<int>(mesh.get_index(level, ii  , j));
            // bool exists_i_jp1;
            // bool exists_i_jm1;
            // bool exists_ip1_j;
            // bool exists_im1_j;
            auto i_jp1 = static_cast<int>(mesh.get_index(level, ii  , j+1));
            auto i_jm1 = static_cast<int>(mesh.get_index(level, ii  , j-1));
            auto ip1_j = static_cast<int>(mesh.get_index(level, ii+1, j));
            auto im1_j = static_cast<int>(mesh.get_index(level, ii-1, j));
            MatSetValue(A, i_j, i_j  , v_diag, INSERT_VALUES);
            MatSetValue(A, i_j, i_jp1,  v_off, INSERT_VALUES);
            MatSetValue(A, i_j, i_jm1,  v_off, INSERT_VALUES);
            MatSetValue(A, i_j, ip1_j,  v_off, INSERT_VALUES);
            MatSetValue(A, i_j, im1_j,  v_off, INSERT_VALUES);
        }
    });

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();

    // Projection
    for(std::size_t level=min_level; level<max_level; ++level)
    {
        auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                         mesh[mesh_id_t::cells][level+1])
                    .on(level);

        set([&](const auto& i, const auto& index)
        {
            auto j = index[0];

            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = static_cast<int>(mesh.get_index(level, ii, j));
                MatSetValue(A, i_cell, i_cell, 1, INSERT_VALUES);

                auto i1 = static_cast<int>(mesh.get_index(level + 1,     2*ii,     2*j));
                auto i2 = static_cast<int>(mesh.get_index(level + 1, 2*ii + 1,     2*j));
                auto i3 = static_cast<int>(mesh.get_index(level + 1,     2*ii, 2*j + 1));
                auto i4 = static_cast<int>(mesh.get_index(level + 1, 2*ii + 1, 2*j + 1));
                MatSetValue(A, i_cell, i1, -0.25, INSERT_VALUES);
                MatSetValue(A, i_cell, i2, -0.25, INSERT_VALUES);
                MatSetValue(A, i_cell, i3, -0.25, INSERT_VALUES);
                MatSetValue(A, i_cell, i4, -0.25, INSERT_VALUES);
            }
        });
    }

    // Prediction (order 1)
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
                double v = 1.;
                auto i_cell = static_cast<int>(mesh.get_index(level, ii, j));
                MatSetValue(A, i_cell, i_cell, v, INSERT_VALUES);

                int sign_i = (ii & 1)? -1: 1;

                for(int is = -1; is<2; ++is)
                {
                    auto i1 = static_cast<int>(mesh.get_index(level - 1, (ii>>1), (j>>1) + is));
                    v = -sign_j*pred[is + 1];
                    MatSetValue(A, i_cell, i1, v, INSERT_VALUES);

                    i1 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) + is, (j>>1)));
                    v = -sign_i*pred[is + 1];
                    MatSetValue(A, i_cell, i1, v, INSERT_VALUES);
                }

                auto i1 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) - 1));
                auto i2 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) - 1));
                auto i3 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) + 1));
                auto i4 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) + 1));

                v = sign_i*sign_j*pred[0]*pred[0];
                MatSetValue(A, i_cell, i1, v, INSERT_VALUES);
                v = sign_i*sign_j*pred[2]*pred[0];
                MatSetValue(A, i_cell, i2, v, INSERT_VALUES);
                v = sign_i*sign_j*pred[0]*pred[2];
                MatSetValue(A, i_cell, i3, v, INSERT_VALUES);
                v = sign_i*sign_j*pred[2]*pred[2];
                MatSetValue(A, i_cell, i4, v, INSERT_VALUES);

                auto i0 = static_cast<int>(mesh.get_index(level - 1, (ii>>1), (j>>1)));
                MatSetValue(A, i_cell, i0, -1., INSERT_VALUES);
            }
        });
    }

    // Boundary:
    // First, this sets the b.c. to full Neumann.
    //xt::xtensor_fixed<int, xt::xshape<4, 2>> stencils{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    double penalty_coeff = 1000;
    for(std::size_t level=min_level; level<=max_level; ++level)
    {
        samurai_new::boundary(mesh, level, 
        [&] (const auto& i, const auto& index, const auto& out_vect)
        {
            auto j = index[0];

            if (out_vect[0] != 0 && out_vect[1] != 0) // corners
            {
                for(int ii=i.start; ii<i.end; ++ii)
                {
                    auto i_out   = static_cast<int>(mesh.get_index(level, ii, j));
                    MatSetValue(A, i_out, i_out, 1, INSERT_VALUES);
                }
            }
            else // Cartesian direction
            {
                bool neumann = false;
                if (neumann || dirichlet_enfcmt == Penalization)
                {
                    double v_off = off_diag_coeff_2D(level);
                    double v_diag = -v_off;
                    if (dirichlet_enfcmt == Penalization)
                        v_diag *= (1 + penalty_coeff);
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_out = static_cast<int>(mesh.get_index(level, ii              , j              ));
                        auto i_in  = static_cast<int>(mesh.get_index(level, ii - out_vect[0], j - out_vect[1]));
                        MatSetValue(A, i_out, i_out, v_diag, INSERT_VALUES);
                        MatSetValue(A, i_out, i_in ,  v_off, INSERT_VALUES);
                    }
                }
                else if (dirichlet_enfcmt == Elimination)
                {
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_out = static_cast<int>(mesh.get_index(level, ii              , j              ));
                        auto i_in  = static_cast<int>(mesh.get_index(level, ii - out_vect[0], j - out_vect[1]));
                        MatSetValue(A, i_out, i_out, 1, INSERT_VALUES);
                        MatSetValue(A, i_in , i_out, 0, INSERT_VALUES); // Remove the coefficient that was added before
                    }
                }
                else if (dirichlet_enfcmt == OnesOnDiagonal)
                {
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_out   = static_cast<int>(mesh.get_index(level, ii, j));
                        MatSetValue(A, i_out, i_out, 1, INSERT_VALUES);
                    }
                }
            }
        });
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
    /*double penalty_coeff = 1000;
    for(std::size_t level=min_level; level<=max_level; ++level)
    {
        // for(std::size_t is = 0; is < stencils.shape()[0]; ++is)
        // {
        //     auto s = xt::view(stencils, is);
        //     auto set = samurai::difference(samurai::translate(mesh[mesh_id_t::cells_and_ghosts][level], s),
        //                                 mesh.domain()).on(level);
            auto set = samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level],
                                         mesh.domain()).on(level);

            set([&](const auto& i, const auto& index)
            {
                auto j = index[0];
                if (dirichlet_enfcmt == Penalization)
                {
                    double v_diag, v_off;
                    coeffs_2D(level, v_diag, v_off);
                    double v_diag *= (1 + penalty_coeff);
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_out = static_cast<int>(mesh.get_index(level, ii, j));
                        MatSetValue(A, i_out, i_out, v_diag, INSERT_VALUES);
                    }
                }
                else if (dirichlet_enfcmt == Elimination)
                {
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_j   = static_cast<int>(mesh.get_index(level, ii  , j  ));
                        auto i_jp1 = static_cast<int>(mesh.get_index(level, ii  , j+1));
                        auto i_jm1 = static_cast<int>(mesh.get_index(level, ii  , j-1));
                        auto ip1_j = static_cast<int>(mesh.get_index(level, ii+1, j  ));
                        auto im1_j = static_cast<int>(mesh.get_index(level, ii-1, j  ));
                            MatSetValue(A, i_j  , i_j  , 1, INSERT_VALUES);
                        if (ip1_j < n) 
                        {
                            MatSetValue(A, i_j  , ip1_j, 0, INSERT_VALUES);
                            MatSetValue(A, ip1_j, i_j  , 0, INSERT_VALUES);
                        }
                        if (im1_j < n) 
                        {
                            MatSetValue(A, i_j  , im1_j, 0, INSERT_VALUES);
                            MatSetValue(A, im1_j, i_j  , 0, INSERT_VALUES);
                        }
                        if (i_jp1 < n) 
                        {
                            MatSetValue(A, i_j  , i_jp1, 0, INSERT_VALUES);
                            MatSetValue(A, i_jp1, i_j  , 0, INSERT_VALUES);
                        }
                        if (i_jm1 < n) 
                        {
                            MatSetValue(A, i_j  , i_jm1, 0, INSERT_VALUES);
                            MatSetValue(A, i_jm1, i_j  , 0, INSERT_VALUES);
                        }
                    }
                }
                else if (dirichlet_enfcmt == OnesOnDiagonal)
                {
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        // auto i_out = static_cast<int>(mesh.get_index(level, ii, j));
                        // auto i_in  = static_cast<int>(mesh.get_index(level, ii - s[0], j - s[1]));
                        // MatSetValue(A, i_out, i_out, 1., INSERT_VALUES);
                        // MatSetValue(A, i_out, i_in , 0., INSERT_VALUES);

                        auto i_j   = static_cast<int>(mesh.get_index(level, ii  , j  ));
                        auto i_jp1 = static_cast<int>(mesh.get_index(level, ii  , j+1));
                        auto i_jm1 = static_cast<int>(mesh.get_index(level, ii  , j-1));
                        auto ip1_j = static_cast<int>(mesh.get_index(level, ii+1, j  ));
                        auto im1_j = static_cast<int>(mesh.get_index(level, ii-1, j  ));
                                       MatSetValue(A, i_j, i_j  , 1, INSERT_VALUES);
                        if (ip1_j < n) MatSetValue(A, i_j, ip1_j, 0, INSERT_VALUES);
                        if (im1_j < n) MatSetValue(A, i_j, im1_j, 0, INSERT_VALUES);
                        if (i_jp1 < n) MatSetValue(A, i_j, i_jp1, 0, INSERT_VALUES);
                        if (i_jm1 < n) MatSetValue(A, i_j, i_jm1, 0, INSERT_VALUES);
                    }
                }
            });
        //}
    }*/

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
    //using interval_t = typename Mesh::interval_t;

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