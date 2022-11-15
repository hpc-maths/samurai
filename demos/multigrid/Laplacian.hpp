#pragma once
#include "Laplacian1D.cpp"
#include "Laplacian2D.cpp"
#include "Laplacian3D.cpp"
#include "samurai_new/petsc_assembly.hpp"
#include "samurai_new/gauss_legendre.hpp"
#include "samurai_new/boundary.hpp"


// constexpr power function
template <typename T>
constexpr T ce_pow(T num, unsigned int pow)
{
    return pow == 0 ? 1 : num * ce_pow(num, pow-1);
}




template<class Field>
class Laplacian
{
public:
    using field_t = Field;
    using Mesh = typename Field::mesh_t;
    using mesh_id_t = typename Mesh::mesh_id_t;
    static constexpr std::size_t dim = Field::dim;

    Mesh& mesh;

    Laplacian(Mesh& m) :
        mesh(m)
    {}

    /**
     * @brief Creates a coarse object from a coarse mesh and a fine object.
     * @note  This method is used by the multigrid.
    */
    static Laplacian create_coarse(const Laplacian& /*fine*/, Mesh& coarse_mesh)
    {
        return Laplacian(coarse_mesh);
    }

    /**
     * @brief Performs the memory preallocation of the Petsc matrix.
     * @see assemble_matrix
    */
    void create_matrix(Mat& A)
    {
        auto n = static_cast<PetscInt>(mesh.nb_cells());

        MatCreate(PETSC_COMM_SELF, &A);
        MatSetSizes(A, n, n, n, n);
        MatSetFromOptions(A);

        MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, sparsity_pattern().data());
    }

    /**
     * @brief Inserts the coefficent into a preallocated matrix and performs the assembly.
    */
    void assemble_matrix(Mat& A)
    {
        assemble_scheme_on_uniform_grid(A);
        assemble_projection(A);
        assemble_prediction(A);

        // Flags the matrix as symmetric positive-definite.
        MatSetOption(A, MAT_SPD, PETSC_TRUE);

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    }

    /**
     * @brief Creates a right-hand side in the form of a Field.
     * @param source_function Source function of the diffusion problem (must return double).
     * @param source_poly_degree Polynomial degree of the source function (use -1 if it is not a polynomial function)
     * @note Sets homogeneous Dirichlet boundary condition. For non-homogeneous condition, use enforce_dirichlet_bc().
    */
    template<class Func>
    Field create_rhs(Func&& source_function, int source_poly_degree=-1)
    {
        Field rhs("rhs", mesh);
        rhs.fill(0);
        samurai_new::GaussLegendre gl(source_poly_degree);

        samurai::for_each_cell(mesh, [&](const auto& cell)
        {
            const double& h = cell.length;
            rhs.array()[cell.index] = gl.quadrature(cell, source_function) / pow(h, dim);
        });
        return rhs;
    }

    /**
     * @brief Enforces Dirichlet boundary condition (on the whole boundary).
     * @param rhs_field rhs_field to update (created by create_rhs()).
     * @param dirichlet Dirichlet function: takes coordinates and returns double.
     *                  It will be called with coordinates on the boundary.
    */
    template<class Func>
    void enforce_dirichlet_bc(Field& rhs_field, Func&& dirichlet)
    {
        if (&rhs_field.mesh() != &mesh)
            assert(false && "Not the same mesh");

        using coord_index_t = typename Mesh::interval_t::coord_index_t;

        samurai_new::in_boundary(mesh, FV_stencil(),
        [&] (const auto& mesh_interval, const auto& towards_bdry)
        {
            samurai_new::for_each_cell(mesh, mesh_interval.level, mesh_interval.i, mesh_interval.index, [&](const samurai::Cell<coord_index_t, dim>& cell)
            {
                double h = cell.length;
                auto boundary_point = cell.face_center(towards_bdry);
                auto dirichlet_value = dirichlet(boundary_point);
                rhs_field.array()[cell.index] += 2/(h*h) * dirichlet_value;
            });
        });
    }

private:
    /**
     * Set useful sizes to define the sparsity pattern of the matrix and perform the preallocation.
     * @see samurai_new::petsc::nnz_per_row<cfg>(mesh), used in sparsity_pattern()
    */
    using cfg = samurai_new::petsc::PetscAssemblyConfig
    <
        // ----  Stencil size 
        // Cell-centered Finite Volume scheme:
        // center + 1 neighbour in each Cartesian direction (2*dim directions) --> 1+2=3 in 1D
        //                                                                         1+4=5 in 2D
        1 + 2*dim,

        // ----  Projection stencil size
        // cell + 2^dim children --> 1+2=3 in 1D 
        //                           1+4=5 in 2D
        1 + (1 << dim), 

        // ----  Prediction stencil size
        // Here, order 1:
        // cell + hypercube of 3 cells --> 1+3= 4 in 1D
        //                                 1+9=10 in 2D
        1 + ce_pow(3, dim), 

        // ---- Index of the stencil center
        // (as defined in FV_stencil())
        1, 

        // ---- Start index and size of contiguous cell indices
        // (as defined in FV_stencil())
        // Here, [left, center, right].
        0, 3
    >;


    /**
     * @return the stencil of the Finite Volume scheme.
    */
    inline samurai_new::StencilShape<dim, cfg::scheme_stencil_size> FV_stencil()
    {
        static_assert(dim >= 1 || dim <= 3, "Finite Volume stencil not implemented for this dimension");

        if constexpr (dim == 1)
        {
            // 3-point stencil:
            //    left, center, right
            return {{-1}, {0}, {1}};
        }
        else if constexpr (dim == 2)
        {
            // 5-point stencil:
            //       left,   center,  right,   bottom,  top 
            return {{-1, 0}, {0, 0},  {1, 0}, {0, -1}, {0, 1}};
        }
        else if constexpr (dim == 3)
        {
            // 7-point stencil:
            //       left,   center,    right,   front,    back,    bottom,    top
            return {{-1,0,0}, {0,0,0},  {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
        }
        return samurai_new::StencilShape<dim, cfg::scheme_stencil_size>();
    }

    std::array<double, cfg::scheme_stencil_size> FV_coefficients(double h)
    {
        double one_over_h2 = 1/(h*h);

        std::array<double, cfg::scheme_stencil_size> coeffs;
        for (unsigned int i = 0; i<cfg::scheme_stencil_size; ++i)
        {
            coeffs[i] = -one_over_h2;
        }
        coeffs[cfg::center_index] = (cfg::scheme_stencil_size-1) * one_over_h2;
        return coeffs;
    }

    /**
     * @brief sparsity pattern of the matrix
     * @return vector that stores, for each row index in the matrix, the number of non-zero coefficients.
    */
    std::vector<PetscInt> sparsity_pattern()
    {
        // Scheme + projection + prediction
        std::vector<PetscInt> nnz = samurai_new::petsc::nnz_per_row<cfg>(mesh);
        return nnz;
    }

    /**
     * @brief Inserts coefficients into the matrix.
     * This function defines the scheme on a uniform, Cartesian grid.
    */
    void assemble_scheme_on_uniform_grid(Mat& A)
    {
        samurai_new::petsc::set_coefficients<cfg>(A, mesh, FV_stencil(), [&] (double h)
        {
            return FV_coefficients(h);
        });
    }

    /**
     * @brief Inserts the coefficients corresponding the projection operator into the matrix.
    */
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

    /**
     * @brief Inserts the coefficients corresponding the prediction operator into the matrix.
    */
    void assemble_prediction(Mat& A)
    {
        assemble_prediction_impl(std::integral_constant<std::size_t, dim>{}, A, mesh);
    }
};

