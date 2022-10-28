#pragma once
#include "samurai_new/multigrid/petsc/utils.hpp"
#include "utils.hpp"
#include "samurai_new/boundary.hpp"
#include "samurai_new/indices.hpp"

//-------------------//
//     Laplacian     //
// Implementation 1D //
//-------------------//

template<class Mesh>
PetscErrorCode assemble_matrix_impl(std::integral_constant<std::size_t, 1>, Mat& A, Mesh& mesh, DirichletEnforcement dirichlet_enfcmt)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    static constexpr std::size_t dim = Mesh::dim;

    // Finite Volume cell-centered scheme: 3-point stencil in 1D, 5-point stencil in 2D, etc.
    // center + 1 neighbour in each Cartesian direction (2*dim directions)
    static constexpr PetscInt scheme_stencil_size = 1 + 2*dim;

    static constexpr PetscInt number_of_children = (1 << dim);

    // For each group of cells given by stencil_shape, we want to capture the indices as PetscInt.
    // 3-point stencil:                                                   center, left, right
    samurai_new::StencilIndices<PetscInt, dim, scheme_stencil_size> stencil({{0}, {-1}, {1}});

    samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
    {
        double one_over_h2 = 1/(h*h);

        double coeffs_stencil_row[3];
        coeffs_stencil_row[0] =                          -one_over_h2;
        coeffs_stencil_row[1] = (scheme_stencil_size-1) * one_over_h2;
        coeffs_stencil_row[2] =                          -one_over_h2;

        samurai_new::for_each_stencil<PetscInt>(mesh, mesh[mesh_id_t::cells], level, stencil,
        [&] (const std::array<PetscInt, scheme_stencil_size>& indices)
        {
            auto& center = indices[0];
            auto& left   = indices[1];
            auto& right  = indices[2];

            PetscInt stencil_row[3];
            stencil_row[0] = left;
            stencil_row[1] = center;
            stencil_row[2] = right;
            MatSetValues(A, 1, &center, 3, stencil_row, coeffs_stencil_row, INSERT_VALUES);
        });
    });

    // Projection
    samurai_new::for_each_cell_and_children<PetscInt>(mesh, 
    [&] (PetscInt cell, const std::array<PetscInt, number_of_children>& children)
    {
        MatSetValue(A, cell, cell, 1, INSERT_VALUES);
        for (unsigned int i=0; i<number_of_children; ++i)
        {
            MatSetValue(A, cell, children[i], -1./number_of_children, INSERT_VALUES);
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

            set([&](const auto& i, const auto&)
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
    double penalty_coeff = 1000;
    for(std::size_t level=min_level; level<=max_level; ++level)
    {
        for(std::size_t is = 0; is < stencils.shape()[0]; ++is)
        {
            auto s = xt::view(stencils, is);
            auto set = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], s),
                                        mesh.domain()).on(level);

            set([&](const auto& i, const auto&)
            {   
                if (dirichlet_enfcmt == Penalization)
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
Vec assemble_rhs_impl(std::integral_constant<std::size_t, 1>, Field& rhs_field, DirichletEnforcement /*dirichlet_enfcmt*/)
{
    using Mesh = typename Field::mesh_t;
    using mesh_id_t = typename Mesh::mesh_id_t;
    //using interval_t = typename Mesh::interval_t;

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