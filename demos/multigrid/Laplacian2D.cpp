#pragma once
#include "samurai_new/multigrid/petsc/utils.hpp"

//-------------------//
//     Laplacian     //
// Implementation 2D //
//-------------------//

template<class Mesh>
std::vector<int> preallocate_matrix_impl(std::integral_constant<std::size_t, 2>, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    std::size_t n = mesh.nb_cells();

    std::vector<int> nnz(n, 1);

    samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
    {
        auto j = index[0];
        for(int ii=i.start; ii<i.end; ++ii)
        {
            auto i_j = static_cast<int>(mesh.get_index(level, ii, j));
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
                auto i_cell = static_cast<int>(mesh.get_index(level, ii, j));
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
                auto i_cell = static_cast<int>(mesh.get_index(level, ii, j));
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
                auto i_cell = static_cast<int>(mesh.get_index(level, ii, j));
                nnz[i_cell] = 2;
            }
        });
    }

    return nnz;
}

template<class Mesh>
PetscErrorCode assemble_matrix_impl(std::integral_constant<std::size_t, 2>, Mat& A, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    using interval_t = typename Mesh::interval_t;

    std::size_t n = mesh.nb_cells();

    for(int i=0; i<n; ++i)
    {
        double v = 1;
        MatSetValues(A, 1, &i, 1, &i, &v, INSERT_VALUES);
    }

    samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
    {
        auto j = index[0];

        double dx = 1./(1<<level);
        double one_over_dx2 = 1./(dx*dx);

        double v_diag = -4*one_over_dx2;
        double v_off = one_over_dx2;
        for(int ii=i.start; ii<i.end; ++ii)
        {
            auto i_j = static_cast<int>(mesh.get_index(level, ii, j));
            auto i_jp1 = static_cast<int>(mesh.get_index(level, ii, j+1));
            auto i_jm1 = static_cast<int>(mesh.get_index(level, ii, j-1));
            auto ip1_j = static_cast<int>(mesh.get_index(level, ii+1, j));
            auto im1_j = static_cast<int>(mesh.get_index(level, ii-1, j));
            MatSetValues(A, 1, &i_j, 1, &i_j, &v_diag, INSERT_VALUES);
            MatSetValues(A, 1, &i_j, 1, &ip1_j, &v_off, INSERT_VALUES);
            MatSetValues(A, 1, &i_j, 1, &im1_j, &v_off, INSERT_VALUES);
            MatSetValues(A, 1, &i_j, 1, &i_jp1, &v_off, INSERT_VALUES);
            MatSetValues(A, 1, &i_j, 1, &i_jm1, &v_off, INSERT_VALUES);
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
                double v = 1;
                auto i_cell = static_cast<int>(mesh.get_index(level, ii, j));
                MatSetValues(A, 1, &i_cell, 1, &i_cell, &v, INSERT_VALUES);

                v = -0.25;
                auto i1 = static_cast<int>(mesh.get_index(level + 1,     2*ii,     2*j));
                auto i2 = static_cast<int>(mesh.get_index(level + 1, 2*ii + 1,     2*j));
                auto i3 = static_cast<int>(mesh.get_index(level + 1,     2*ii, 2*j + 1));
                auto i4 = static_cast<int>(mesh.get_index(level + 1, 2*ii + 1, 2*j + 1));
                MatSetValues(A, 1, &i_cell, 1, &i1, &v, INSERT_VALUES);
                MatSetValues(A, 1, &i_cell, 1, &i2, &v, INSERT_VALUES);
                MatSetValues(A, 1, &i_cell, 1, &i3, &v, INSERT_VALUES);
                MatSetValues(A, 1, &i_cell, 1, &i4, &v, INSERT_VALUES);
            }
        });
    }

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
                MatSetValues(A, 1, &i_cell, 1, &i_cell, &v, INSERT_VALUES);

                int sign_i = (ii & 1)? -1: 1;

                for(int is = -1; is<2; ++is)
                {
                    auto i1 = static_cast<int>(mesh.get_index(level - 1, (ii>>1), (j>>1) + is));
                    v = -sign_j*pred[is + 1];
                    MatSetValues(A, 1, &i_cell, 1, &i1, &v, INSERT_VALUES);

                    i1 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) + is, (j>>1)));
                    v = -sign_i*pred[is + 1];
                    MatSetValues(A, 1, &i_cell, 1, &i1, &v, INSERT_VALUES);
                }

                auto i1 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) - 1));
                auto i2 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) - 1));
                auto i3 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) + 1));
                auto i4 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) + 1));

                v = sign_i*sign_j*pred[0]*pred[0];
                MatSetValues(A, 1, &i_cell, 1, &i1, &v, INSERT_VALUES);
                v = sign_i*sign_j*pred[2]*pred[0];
                MatSetValues(A, 1, &i_cell, 1, &i2, &v, INSERT_VALUES);
                v = sign_i*sign_j*pred[0]*pred[2];
                MatSetValues(A, 1, &i_cell, 1, &i3, &v, INSERT_VALUES);
                v = sign_i*sign_j*pred[2]*pred[2];
                MatSetValues(A, 1, &i_cell, 1, &i4, &v, INSERT_VALUES);

                auto i0 = static_cast<int>(mesh.get_index(level - 1, (ii>>1), (j>>1)));
                v = -1.;
                MatSetValues(A, 1, &i_cell, 1, &i0, &v, INSERT_VALUES);
            }
        });
    }

    // set boundary for phi
    xt::xtensor_fixed<int, xt::xshape<4, 2>> stencils{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {

        for(std::size_t is = 0; is < stencils.shape()[0]; ++is)
        {
            auto s = xt::view(stencils, is);
            auto set = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], s),
                                        mesh.domain()).on(level);

            set([&](const auto& i, const auto& index)
            {
                auto j = index[0];
                for(int ii=i.start; ii<i.end; ++ii)
                {
                    auto i_out = static_cast<int>(mesh.get_index(level, ii, j));
                    auto i_in = static_cast<int>(mesh.get_index(level, ii - s[0], j - s[1]));
                    double v = 0.5;
                    MatSetValues(A, 1, &i_out, 1, &i_out, &v, INSERT_VALUES);
                    MatSetValues(A, 1, &i_out, 1, &i_in, &v, INSERT_VALUES);
                }
            });
        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    PetscFunctionReturn(0);
}


template<class Field>
Vec assemble_rhs_impl(std::integral_constant<std::size_t, 2>, Field& rhs_field)
{
    using Mesh = typename Field::mesh_t;
    using mesh_id_t = typename Mesh::mesh_id_t;
    using interval_t = typename Mesh::interval_t;

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