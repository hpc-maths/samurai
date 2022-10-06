#pragma once
#include "Laplacian1D.cpp"
#include "Laplacian2D.cpp"

template<class Field>
class Laplacian
{
private:
    bool _eliminate_dirichlet_values;
public:
    using field_t = Field;
    using Mesh = typename Field::mesh_t;
    //static constexpr std::size_t dim = Field::dim;
    const Mesh& mesh;

    Laplacian(const Mesh& m, bool eliminate_dirichlet_values) :
        mesh(m)
    {
        _eliminate_dirichlet_values = eliminate_dirichlet_values;
    }

    static Laplacian create_coarse(const Laplacian& fine, const Mesh& coarse_mesh)
    {
        return Laplacian(coarse_mesh, fine._eliminate_dirichlet_values);
    }

    void create_matrix(Mat& A)
    {
        std::size_t n = mesh.nb_cells();

        MatCreate(PETSC_COMM_SELF, &A);
        MatSetSizes(A, n, n, n, n);
        MatSetFromOptions(A);

        MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, preallocate_matrix_impl(std::integral_constant<std::size_t, Field::dim>{}, mesh).data());
        // ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
    }

    PetscErrorCode assemble_matrix(Mat& A)
    {
        return assemble_matrix_impl(std::integral_constant<std::size_t, Field::dim>{}, A, mesh);
    }

    Vec assemble_rhs(Field& rhs_field)
    {
        if (&rhs_field.mesh() != &mesh)
            assert(false && "Not the same mesh");
        return assemble_rhs_impl(std::integral_constant<std::size_t, Field::dim>{}, rhs_field);
    }
};

