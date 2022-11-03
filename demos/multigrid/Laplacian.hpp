#pragma once
#include "Laplacian1D.cpp"
#include "Laplacian2D.cpp"
#include "Laplacian3D.cpp"
#include "samurai_new/petsc_assembly.hpp"


// constexpr power function
template <typename T>
constexpr T ce_pow(T num, unsigned int pow)
{
    return pow == 0 ? 1 : num * ce_pow(num, pow-1);
}

template<class Field>
class Laplacian
{
private:
    DirichletEnforcement _dirichlet_enfcmt = OnesOnDiagonal;
public:
    using field_t = Field;
    using Mesh = typename Field::mesh_t;
    using mesh_id_t = typename Mesh::mesh_id_t;
    static constexpr std::size_t dim = Field::dim;

    Mesh& mesh;

    Laplacian(Mesh& m, DirichletEnforcement dirichlet_enfcmt) :
        mesh(m)
    {
        _dirichlet_enfcmt = dirichlet_enfcmt;
    }

    static Laplacian create_coarse(const Laplacian& fine, Mesh& coarse_mesh)
    {
        return Laplacian(coarse_mesh, fine._dirichlet_enfcmt);
    }

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
        assemble_scheme_on_uniform_grid(A);
        assemble_projection(A);
        assemble_prediction(A);
        assemble_boundary_condition(A);

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        PetscFunctionReturn(0);
    }

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



    template<class Func>
    void enforce_dirichlet_bc(Field& rhs_field, Func&& dirichlet)
    {
        if (&rhs_field.mesh() != &mesh)
            assert(false && "Not the same mesh");

        samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
        {
            double one_over_h2 = 1/(h*h);

            samurai_new::out_boundary(mesh, level, 
            [&] (const auto& i, const auto& index, const auto& out_vect)
            {
                if (samurai_new::is_cartesian_direction(out_vect))
                {
                    samurai_new::StencilCells<Mesh, 2> out_in_cells(out_in_stencil(out_vect));

                    samurai_new::for_each_stencil(mesh, level, i, index, out_in_cells, 
                    [&] (const auto& cells)
                    {
                        auto& in_cell  = cells[1];

                        // translate center by h/2
                        auto bdry_point = in_cell.center() + (h/2)* out_vect;

                        rhs_field.array()[in_cell.index] += 2 * one_over_h2 * dirichlet(bdry_point);
                    });
                }
            });
        });
    }

private:

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

    // Stencil size on outer boundary cells (in the Cartesian directions)
    static constexpr PetscInt cart_bdry_stencil_size = 2;
    // Stencil size on outer boundary cells (in the diagonal directions)
    static constexpr PetscInt diag_bdry_stencil_size = 1;

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

    std::vector<PetscInt> sparsity_pattern()
    {
        // Scheme, projection, prediction
        std::vector<PetscInt> nnz = samurai_new::petsc::nnz_per_row<cfg>(mesh);

        // Boundary conditions
        samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double)
        {
            samurai_new::out_boundary(mesh, level, 
            [&] (const auto& i, const auto& index, const auto& out_vect)
            {
                PetscInt n_coeffs = samurai_new::is_cartesian_direction(out_vect) ? cart_bdry_stencil_size : diag_bdry_stencil_size;
                samurai_new::for_each_cell<std::size_t>(mesh, level, i, index, 
                [&] (std::size_t i_out)
                {
                    nnz[i_out] = n_coeffs;
                });
            });
        });

        return nnz;
    }

    void assemble_scheme_on_uniform_grid(Mat& A)
    {
        static constexpr PetscInt stencil_size = cfg::scheme_stencil_size;
        static constexpr PetscInt stencil_center = cfg::center_index;

        samurai_new::petsc::set_coefficients<cfg>(A, mesh, FV_stencil(), 
        [&] (double h)
        {
            double one_over_h2 = 1/(h*h);

            std::array<double, stencil_size> coeffs;
            for (unsigned int i = 0; i<stencil_size; ++i)
            {
                coeffs[i] = -one_over_h2;
            }
            coeffs[stencil_center] = (stencil_size-1) * one_over_h2;
            return coeffs;
        });
    }

    void assemble_boundary_condition(Mat& A)
    {
        samurai::for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
        {
            double one_over_h2 = 1/(h*h);

            samurai_new::out_boundary(mesh, level, 
            [&] (const auto& i, const auto& index, const auto& out_vect)
            {
                samurai_new::StencilIndices<PetscInt, dim, 2> out_in_indices(out_in_stencil(out_vect));
                bool is_cartesian_direction = samurai_new::is_cartesian_direction(out_vect);
                auto n_zeros = samurai_new::number_of_zeros(out_vect);
                double in_diag_value = (cfg::scheme_stencil_size-1) + dim - n_zeros;
                in_diag_value *= one_over_h2;

                samurai_new::for_each_stencil<PetscInt>(mesh, level, i, index, out_in_indices, 
                [&] (const std::array<PetscInt, 2>& indices)
                {
                    auto& out_cell = indices[0];
                    auto& in_cell  = indices[1];
                    // The outer unknown is eliminated from the system:
                    MatSetValue(A, out_cell, out_cell,             1, INSERT_VALUES);
                    MatSetValue(A,  in_cell, in_cell , in_diag_value, INSERT_VALUES);
                    // The coefficient that was added before (via the interior stencil) is removed.
                    // Note that if out_vect is not a Cartesian direction (out_cell is a corner), then out_cell is not in the stencil of in_cell, so nothing to remove.
                    if (is_cartesian_direction) 
                    {  
                        MatSetValue(A,  in_cell, out_cell,         0, INSERT_VALUES); 
                    }
                });
            });
        });

        if (_dirichlet_enfcmt != OnesOnDiagonal)
            MatSetOption(A, MAT_SPD, PETSC_TRUE);
    }

    template<class Vector>
    samurai_new::StencilShape<dim, 2> out_in_stencil(const Vector& out_vect)
    {
        static_assert(dim >= 1 || dim <= 3, "out_in_stencil() not implemented for this dimension");

        if constexpr (dim == 1)
        {   //   out_cell,  in_cell
            return {{0}, {-out_vect[0]}};
        }
        else if constexpr (dim == 2)
        {   //     out_cell,           in_cell
            return {{0, 0}, {-out_vect[0], -out_vect[1]}};
        }
        else if constexpr (dim == 3)
        {   //      out_cell,                   in_cell
            return {{0, 0, 0}, {-out_vect[0], -out_vect[1], -out_vect[2]}};
        }
        return samurai_new::StencilShape<dim, 2>();
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

    void assemble_prediction(Mat& A)
    {
        assemble_prediction_impl(std::integral_constant<std::size_t, dim>{}, A, mesh);
    }
};

