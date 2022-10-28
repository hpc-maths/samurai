#pragma once
#include "Laplacian1D.cpp"
#include "Laplacian2D.cpp"

template<class Field>
class Laplacian
{
private:
    DirichletEnforcement _dirichlet_enfcmt = OnesOnDiagonal;
public:
    using field_t = Field;
    using Mesh = typename Field::mesh_t;
    static constexpr std::size_t dim = Field::dim;
    const Mesh& mesh;

    Laplacian(const Mesh& m, DirichletEnforcement dirichlet_enfcmt) :
        mesh(m)
    {
        _dirichlet_enfcmt = dirichlet_enfcmt;
    }

    static Laplacian create_coarse(const Laplacian& fine, const Mesh& coarse_mesh)
    {
        return Laplacian(coarse_mesh, fine._dirichlet_enfcmt);
    }

private:
    // constexpr power function
    template <typename T>
    static constexpr T ce_pow(T num, unsigned int pow)
    {
        return pow == 0 ? 1 : num * ce_pow(num, pow-1);
    }
    std::vector<PetscInt> sparsity_pattern()
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        // Finite Volume cell-centered scheme: 3-point stencil in 1D, 5-point stencil in 2D, etc.
        // center + 1 neighbour in each Cartesian direction (2*dim directions)
        static constexpr PetscInt scheme_stencil_size = 1 + 2*dim;
        // Projection:
        // cell + 2^dim children --> 1+2 in 1D, 1+4 in 2D
        static constexpr PetscInt proj_stencil_size   = 1 + (1 << dim);
        // Prediction (order 1): cell + parent's 9-point stencil (in 2D)
        // cell + hypercube of 3 cells --> 1+3 in 1D, 1+9 in 2D
        static constexpr PetscInt pred_stencil_size   = 1 + ce_pow(3, dim);
        // Stencil size on outer boundary cells (in the Cartesian directions)
        static constexpr PetscInt cart_bdry_stencil_size = 2;
        // Stencil size on outer boundary cells (in the diagonal directions)
        static constexpr PetscInt diag_bdry_stencil_size = 1;

        std::size_t n = mesh.nb_cells();

        // Number of non-zeros per row. By default, the stencil size.
        std::vector<PetscInt> nnz(n, scheme_stencil_size);

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
        samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double)
        {
            samurai_new::out_boundary(mesh, level, 
            [&] (const auto& i, const auto& index, const auto& out_vect)
            {
                if (samurai_new::is_cartesian_direction(out_vect))
                {
                    samurai_new::for_each_cell<std::size_t>(mesh, level, i, index, 
                    [&] (std::size_t i_out)
                    {
                        nnz[i_out] = cart_bdry_stencil_size;
                    });
                }
                else
                {
                    samurai_new::for_each_cell<std::size_t>(mesh, level, i, index, 
                    [&] (std::size_t i_out)
                    {
                        nnz[i_out] = diag_bdry_stencil_size;
                    });
                }
            });
        });

        return nnz;
    }

    void assemble_projection(Mat& A)
    {
        static constexpr PetscInt number_of_children = (1 << dim);

        samurai_new::for_each_cell_and_children<PetscInt>(mesh, 
        [&] (PetscInt cell, const std::array<PetscInt, number_of_children>& children)
        {
            MatSetValue(A, cell, cell, 1, INSERT_VALUES);
            for (unsigned int i=0; i<number_of_children; ++i)
            {
                MatSetValue(A, cell, children[i], -1./number_of_children, INSERT_VALUES);
            }
        });
    }

public:
    void create_matrix(Mat& A)
    {
        auto n = static_cast<PetscInt>(mesh.nb_cells());

        MatCreate(PETSC_COMM_SELF, &A);
        MatSetSizes(A, n, n, n, n);
        MatSetFromOptions(A);

        MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, sparsity_pattern().data());
        // MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    }

    PetscErrorCode assemble_matrix(Mat& A)
    {
        assemble_numerical_scheme_impl(std::integral_constant<std::size_t, dim>{}, A, mesh);
        assemble_projection(A);
        assemble_prediction_impl(std::integral_constant<std::size_t, dim>{}, A, mesh);
        assemble_boundary_condition_impl(std::integral_constant<std::size_t, dim>{}, A, mesh, _dirichlet_enfcmt);

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        PetscFunctionReturn(0);
    }

    Vec assemble_rhs(Field& rhs_field)
    {
        if (&rhs_field.mesh() != &mesh)
            assert(false && "Not the same mesh");
        return assemble_rhs_impl(std::integral_constant<std::size_t, Field::dim>{}, rhs_field, _dirichlet_enfcmt);
    }
};

