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
    //static constexpr std::size_t dim = Field::dim;
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

    void create_matrix(Mat& A)
    {
        auto n = static_cast<PetscInt>(mesh.nb_cells());

        MatCreate(PETSC_COMM_SELF, &A);
        MatSetSizes(A, n, n, n, n);
        MatSetFromOptions(A);

        MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, preallocate_matrix_impl(std::integral_constant<std::size_t, Field::dim>{}, mesh, _dirichlet_enfcmt).data());
        // ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
    }

    PetscErrorCode assemble_matrix(Mat& A)
    {
        return assemble_matrix_impl(std::integral_constant<std::size_t, Field::dim>{}, A, mesh, _dirichlet_enfcmt);
    }

    Vec assemble_rhs(Field& rhs_field)
    {
        if (&rhs_field.mesh() != &mesh)
            assert(false && "Not the same mesh");
        return assemble_rhs_impl(std::integral_constant<std::size_t, Field::dim>{}, rhs_field, _dirichlet_enfcmt);
    }
};

