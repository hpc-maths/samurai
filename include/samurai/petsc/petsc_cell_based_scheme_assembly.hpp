#pragma once
#include "petsc_assembly.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "../boundary.hpp"

namespace samurai { namespace petsc
{
    template<class cfg, class Field>
    class PetscCellBasedSchemeAssembly : public PetscAssembly
    {
        using Mesh = typename Field::mesh_t;
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;
        using Stencil = Stencil<cfg::scheme_stencil_size, dim>;
        using GetCoefficientsFunc = std::function<std::array<double, cfg::scheme_stencil_size>(double)>;
    public:
        using PetscAssembly::assemble_matrix;
        Mesh& mesh;
    private:
        Stencil _stencil;
        GetCoefficientsFunc _get_coefficients;
    public:
        PetscCellBasedSchemeAssembly(Mesh& m, Stencil s, GetCoefficientsFunc get_coeffs) :
            mesh(m), _stencil(s), _get_coefficients(get_coeffs)
        {}

    private:
        PetscInt matrix_size() override
        {
            return static_cast<PetscInt>(mesh.nb_cells());
        }


        std::vector<PetscInt> sparsity_pattern() override
        {
            std::size_t n = mesh.nb_cells();

            // Number of non-zeros per row. By default, 1 (for the unused ghosts on the boundary).
            std::vector<PetscInt> nnz(n, 1);

            // Cells
            for_each_cell_index<std::size_t>(mesh, [&](std::size_t cell)
            {
                nnz[cell] = cfg::scheme_stencil_size;
            });

            // Projection
            for_each_cell_having_children<std::size_t>(mesh, [&] (std::size_t cell)
            {
                nnz[cell] = cfg::proj_stencil_size;
            });

            // Prediction
            for_each_cell_having_parent<std::size_t>(mesh, [&] (std::size_t cell)
            {
                nnz[cell] = cfg::pred_stencil_size;
            });

            return nnz;
        }


        void assemble_scheme_on_uniform_grid(Mat& A) override
        {
            //--------------//
            //   Interior   //
            //--------------//

            for_each_stencil<PetscInt>(mesh, _stencil, _get_coefficients,
            [&] (const std::array<PetscInt, cfg::scheme_stencil_size>& indices, const std::array<double, cfg::scheme_stencil_size>& coeffs)
            {
                if constexpr(cfg::contiguous_indices_start > 0)
                {
                    for (unsigned int i=0; i<cfg::contiguous_indices_start; ++i)
                    {
                        MatSetValue(A, indices[cfg::center_index], indices[i], coeffs[i], INSERT_VALUES);
                    }
                }

                if constexpr(cfg::contiguous_indices_size > 0)
                {
                    MatSetValues(A, 1, &indices[cfg::center_index], static_cast<PetscInt>(cfg::contiguous_indices_size), &indices[cfg::contiguous_indices_start], &coeffs[cfg::contiguous_indices_start], INSERT_VALUES);
                }

                if constexpr(cfg::contiguous_indices_start + cfg::contiguous_indices_size < cfg::scheme_stencil_size)
                {
                    for (unsigned int i=cfg::contiguous_indices_start + cfg::contiguous_indices_size; i<cfg::scheme_stencil_size; ++i)
                    {
                        MatSetValue(A, indices[cfg::center_index], indices[i], coeffs[i], INSERT_VALUES);
                    }
                }
            });

            // Must flush to use ADD_VALUES instead of INSERT_VALUES
            MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);


            //--------------//
            //   Boundary   //
            //--------------//

            // 1 - The outside ghosts are 'eliminated' from the system, we simply add 1 on the diagonal.
            for_each_outside_ghost_index<PetscInt>(mesh, [&](PetscInt ghost)
            {
                MatSetValue(A, ghost, ghost, 1, ADD_VALUES);
            });

            // 2 - The (opposite of the) contribution of the outer ghost is added to the diagonal of stencil center
            for_each_stencil_center_and_outside_ghost<PetscInt>(mesh, _stencil, _get_coefficients, [&] (const std::array<PetscInt, 2>& indices, const auto&, double ghost_coeff)
            {
                auto& cell  = indices[0];
                auto& ghost = indices[1];
                MatSetValue(A, cell,  cell, -ghost_coeff, ADD_VALUES); // the coeff is added to the center of the stencil
                MatSetValue(A, cell, ghost, -ghost_coeff, ADD_VALUES); // the coeff of the ghost is removed
            });


            // Must flush to use INSERT_VALUES instead of ADD_VALUES
            MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
        }



        void assemble_projection(Mat& A) override
        {
            static constexpr PetscInt number_of_children = (1 << dim);

            for_each_cell_and_children<PetscInt>(mesh, 
            [&] (PetscInt cell, const std::array<PetscInt, number_of_children>& children)
            {
                MatSetValue(A, cell, cell, 1, INSERT_VALUES);
                for (unsigned int i=0; i<number_of_children; ++i)
                {
                    MatSetValue(A, cell, children[i], -1./number_of_children, INSERT_VALUES);
                }
            });
        }


        
        void assemble_prediction(Mat& A) override
        {
            assemble_prediction_impl(std::integral_constant<std::size_t, dim>{}, A, mesh);
        }


    public:
        /**
         * @brief Creates a right-hand side in the form of a Field.
         * @param name Name of the returned Field.
         * @param f Continuous function (must return double).
         * @param poly_degree Polynomial degree of the function (use -1 if it is not a polynomial function)
         * @note Sets homogeneous Dirichlet boundary condition. For non-homogeneous condition, use enforce_dirichlet_bc().
        */
        template<class Func>
        Field discretize(const std::string& name, Func&& f, int poly_degree=-1)
        {
            Field field(name, mesh);
            field.fill(0);
            GaussLegendre gl(poly_degree);

            for_each_cell(mesh, [&](const auto& cell)
            {
                const double& h = cell.length;
                field[cell] = gl.quadrature(cell, f) / pow(h, dim);
            });
            return field;
        }
    };

}} // end namespace



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
        auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
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
                    double v = -sign_i*pred[static_cast<unsigned int>(is + 1)];
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
        auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
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
                    MatSetValue(A, i_cell, i1, -sign_j*pred[static_cast<unsigned int>(is + 1)], INSERT_VALUES);

                    i1 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) + is, (j>>1)));
                    MatSetValue(A, i_cell, i1, -sign_i*pred[static_cast<unsigned int>(is + 1)], INSERT_VALUES);
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
        auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                mesh[mesh_id_t::cells][level-1])
                .on(level);

        //std::array<double, 3> pred{{1./8, 0, -1./8}};
        set([&](const auto&, const auto&)
        {
            assert(false && "non implemented");
        });
    }
}