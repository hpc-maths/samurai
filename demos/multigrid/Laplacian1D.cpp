#pragma once
#include "samurai_new/multigrid/petsc/utils.hpp"

//-------------------//
//     Laplacian     //
// Implementation 1D //
//-------------------//

template<class Mesh>
std::vector<int> preallocate_matrix_impl(std::integral_constant<std::size_t, 1>, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    std::size_t n = mesh.nb_cells();

    std::vector<int> nnz(n, 1);

    samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& )
    {
        for(int ii=i.start; ii<i.end; ++ii)
        {
            auto i_ = static_cast<int>(mesh.get_index(level, ii));
            nnz[i_] = 3;
        }
    });

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();

    for(std::size_t level=min_level; level<max_level; ++level)
    {
        auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                        mesh[mesh_id_t::cells][level+1])
                    .on(level);

        set([&](const auto& i, const auto& )
        {
            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = static_cast<int>(mesh.get_index(level, ii));
                nnz[i_cell] = 3;
            }
        });
    }

    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {
        auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                        mesh[mesh_id_t::cells][level-1])
                .on(level);

        set([&](const auto& i, const auto&)
        {
            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = static_cast<int>(mesh.get_index(level, ii));
                nnz[i_cell] = 4;
            }
        });
    }

    for(std::size_t level=min_level; level<=max_level; ++level)
    {
        auto set = samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level],
                                    mesh.domain())
                .on(level);

        set([&](const auto& i, const auto&)
        {
            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = static_cast<int>(mesh.get_index(level, ii));
                nnz[i_cell] = 2;
            }
        });
    }
    return nnz;
}

template<class Mesh>
PetscErrorCode assemble_matrix_impl(std::integral_constant<std::size_t, 1>, Mat& A, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    using interval_t = typename Mesh::interval_t;

    std::size_t n = mesh.nb_cells();

    for(int i=0; i<n; ++i)
    {
        MatSetValue(A, i, i, 1., INSERT_VALUES);
    }

    samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto&)
    {
        double dx = 1./(1<<level);
        double one_over_dx2 = 1./(dx*dx);

        double v_diag = 2*one_over_dx2;
        double v_off = -one_over_dx2;
        for(int ii=i.start; ii<i.end; ++ii)
        {
            auto i_ = static_cast<int>(mesh.get_index(level, ii));
            auto ip1_ = static_cast<int>(mesh.get_index(level, ii+1));
            auto im1_ = static_cast<int>(mesh.get_index(level, ii-1));
            MatSetValue(A, i_, i_,  v_diag, INSERT_VALUES);
            MatSetValue(A, i_, ip1_, v_off, INSERT_VALUES);
            MatSetValue(A, i_, im1_, v_off, INSERT_VALUES);
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

        set([&](const auto& i, const auto&)
        {
            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = static_cast<int>(mesh.get_index(level, ii));
                MatSetValue(A, i_cell, i_cell, 1., INSERT_VALUES);

                auto i1 = static_cast<int>(mesh.get_index(level + 1,     2*ii));
                auto i2 = static_cast<int>(mesh.get_index(level + 1, 2*ii + 1));
                MatSetValue(A, i_cell, i1, -0.5, INSERT_VALUES);
                MatSetValue(A, i_cell, i2, -0.5, INSERT_VALUES);
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

    // Boundary:
    // First, this sets the b.c. to full Neumann.
    xt::xtensor_fixed<int, xt::xshape<2, 1>> stencils{{-1}, {1}};
    for(std::size_t level=min_level; level<=max_level; ++level)
    {
        for(std::size_t is = 0; is < stencils.shape()[0]; ++is)
        {
            auto s = xt::view(stencils, is);
            auto set = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], s),
                                        mesh.domain()).on(level);

            set([&](const auto& i, const auto& index)
            {
                double dx = 1./(1<<level);
                double one_over_dx2 = 1./(dx*dx);
                double v_off = -one_over_dx2;
                for(int ii=i.start; ii<i.end; ++ii)
                {
                    auto i_out = static_cast<int>(mesh.get_index(level, ii));
                    auto i_in  = static_cast<int>(mesh.get_index(level, ii - s[0]));
                    MatSetValue(A, i_out, i_out, -v_off, INSERT_VALUES);
                    MatSetValue(A, i_out, i_in ,  v_off, INSERT_VALUES);
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
    bool penalty_method = false;
    double penalty_coeff = 1000;
    for(std::size_t level=min_level; level<=max_level; ++level)
    {
        for(std::size_t is = 0; is < stencils.shape()[0]; ++is)
        {
            auto s = xt::view(stencils, is);
            auto set = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], s),
                                        mesh.domain()).on(level);

            set([&](const auto& i, const auto& index)
            {   
                if (penalty_method)
                {
                    double dx = 1./(1<<level);
                    double one_over_dx2 = 1./(dx*dx);
                    double v_off = -one_over_dx2;
                    double v = -v_off + penalty_coeff*one_over_dx2;
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_out = static_cast<int>(mesh.get_index(level, ii));
                        MatSetValue(A, i_out, i_out, v, INSERT_VALUES);
                    }
                }
                else
                {
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_out = static_cast<int>(mesh.get_index(level, ii));
                        auto i_in = static_cast<int>(mesh.get_index(level, ii - s[0]));
                        MatSetValue(A, i_out, i_out, 1., INSERT_VALUES);
                        MatSetValue(A, i_out, i_in,  0., INSERT_VALUES);
                    }
                }
            });
        }
    }

    MatSetOption(A, MAT_SPD, PETSC_TRUE);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    PetscFunctionReturn(0);
}


template<class Field>
Vec assemble_rhs_impl(std::integral_constant<std::size_t, 1>, Field& rhs_field)
{
    using Mesh = typename Field::mesh_t;
    using mesh_id_t = typename Mesh::mesh_id_t;
    using interval_t = typename Mesh::interval_t;

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