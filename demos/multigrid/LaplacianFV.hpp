#pragma once
#include "samurai_new/petsc_assembly.hpp"
#include "samurai_new/gauss_legendre.hpp"
#include "samurai_new/boundary.hpp"


// constexpr power function
template <typename T>
constexpr T ce_pow(T num, unsigned int pow)
{
    return pow == 0 ? 1 : num * ce_pow(num, pow-1);
}



/**
 * @class LaplacianFV
 * Assemble the matrix for the problem -Lap(u)=f.
 * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
*/
template<class Field>
class LaplacianFV
{
public:
    using field_t = Field;
    using Mesh = typename Field::mesh_t;
    using mesh_id_t = typename Mesh::mesh_id_t;
    static constexpr std::size_t dim = Field::dim;

    Mesh& mesh;

    LaplacianFV(Mesh& m) :
        mesh(m)
    {}

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
    static inline samurai_new::StencilShape<dim, cfg::scheme_stencil_size> FV_stencil()
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

    static std::array<double, cfg::scheme_stencil_size> FV_coefficients(double h)
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

public:
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

        // If the mesh is not refined, then the matrix is flagged as symmetric positive-definite.
        // (The projections/predictions kill the symmetry.)
        PetscBool is_spd = mesh.min_level() == mesh.max_level() ? PETSC_TRUE : PETSC_FALSE;
        MatSetOption(A, MAT_SPD, is_spd);

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
            rhs[cell] = gl.quadrature(cell, source_function) / pow(h, dim);
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
    void enforce_dirichlet_bc(Field& field, Func&& dirichlet)
    {
        if (&field.mesh() != &mesh)
            assert(false && "Not the same mesh");

        using coord_index_t = typename Mesh::interval_t::coord_index_t;

        samurai_new::foreach_cell_on_boundary(mesh, FV_stencil(), FV_coefficients,
        [&] (const samurai::Cell<coord_index_t, dim>& cell, const auto& towards_bdry, double out_coeff)
        {
            auto boundary_point = cell.face_center(towards_bdry);
            auto dirichlet_value = dirichlet(boundary_point);
            field[cell] -= 2 * out_coeff * dirichlet_value;
        });
    }

private:
    /**
     * @brief sparsity pattern of the matrix
     * @return vector that stores, for each row index in the matrix, the number of non-zero coefficients.
    */
    std::vector<PetscInt> sparsity_pattern()
    {
        // Scheme + projection + prediction
        return samurai_new::petsc::nnz_per_row<cfg>(mesh);
    }

    /**
     * @brief Inserts coefficients into the matrix.
     * This function defines the scheme on a uniform, Cartesian grid.
    */
    void assemble_scheme_on_uniform_grid(Mat& A)
    {
        samurai_new::petsc::set_coefficients<cfg>(A, mesh, FV_stencil(), FV_coefficients);
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

public:
    /**
     * @brief Creates a coarse object from a coarse mesh and a fine object.
     * @note  This method is used by the multigrid.
    */
    static LaplacianFV create_coarse(const LaplacianFV& /*fine*/, Mesh& coarse_mesh)
    {
        return LaplacianFV(coarse_mesh);
    }
};


//-----------------------------//
//     Assemble prediction     //
//          (order 1)          //
//-----------------------------//

// 1D

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


// 2D

template<class Mesh>
void assemble_prediction_impl(std::integral_constant<std::size_t, 2>, Mat& A, Mesh& mesh)
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
}


// 3D

template<class Mesh>
void assemble_prediction_impl(std::integral_constant<std::size_t, 3>, Mat& /*A*/, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();
    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {
        auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                         mesh[mesh_id_t::cells][level-1])
                .on(level);

        //std::array<double, 3> pred{{1./8, 0, -1./8}};
        set([&](const auto&, const auto&)
        {
            assert(false && "non implemented");
        });
    }
}