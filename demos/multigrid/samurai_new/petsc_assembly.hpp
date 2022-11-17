#pragma once
#include <petsc.h>
#include <samurai/algorithm.hpp>
#include "indices.hpp"
#include "boundary.hpp"

namespace samurai_new { namespace petsc
{
    class PetscAssembly
    {
    public:
        /**
         * @brief Performs the memory preallocation of the Petsc matrix.
         * @see assemble_matrix
        */
        void create_matrix(Mat& A)
        {
            auto n = matrix_size();

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

            PetscBool is_spd = matrix_is_spd() ? PETSC_TRUE : PETSC_FALSE;
            MatSetOption(A, MAT_SPD, is_spd);

            MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        }

        virtual ~PetscAssembly() {}

    private:
        /**
         * @brief Returns the matrix size.
        */
        virtual PetscInt matrix_size() = 0;

        /**
         * @brief Sparsity pattern of the matrix.
         * @return vector that stores, for each row index in the matrix, the number of non-zero coefficients.
        */
        virtual std::vector<PetscInt> sparsity_pattern() = 0;

        /**
         * @brief Is the matrix symmetric positive-definite?
        */
        virtual bool matrix_is_spd() = 0;

        /**
         * @brief Inserts coefficients into the matrix.
         * This function defines the scheme on a uniform, Cartesian grid.
        */
        virtual void assemble_scheme_on_uniform_grid(Mat& A) = 0;

        /**
         * @brief Inserts the coefficients corresponding to the projection operator into the matrix.
        */
        virtual void assemble_projection(Mat& A) = 0;

        /**
         * @brief Inserts the coefficients corresponding the prediction operator into the matrix.
        */
        virtual void assemble_prediction(Mat& A) = 0;
    };





    template <PetscInt scheme_stencil_size_,
              PetscInt proj_stencil_size_,
              PetscInt pred_stencil_size_,
              PetscInt center_index_,
              PetscInt contiguous_indices_start_,
              PetscInt contiguous_indices_size_>
    struct PetscAssemblyConfig
    {
        static constexpr PetscInt scheme_stencil_size = scheme_stencil_size_;
        static constexpr PetscInt proj_stencil_size = proj_stencil_size_;
        static constexpr PetscInt pred_stencil_size = pred_stencil_size_;
        static constexpr PetscInt center_index = center_index_;
        static constexpr PetscInt contiguous_indices_start = contiguous_indices_start_;
        static constexpr PetscInt contiguous_indices_size = contiguous_indices_size_;
    };

    template<class cfg, class Mesh>
    std::vector<PetscInt> nnz_per_row(const Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        std::size_t n = mesh.nb_cells();

        // Number of non-zeros per row. By default, 1 (for the unused ghosts on the boundary).
        std::vector<PetscInt> nnz(n, 1);

        // Cells
        samurai_new::for_each_cell<std::size_t>(mesh, mesh[mesh_id_t::cells], [&](std::size_t cell)
        {
            nnz[cell] = cfg::scheme_stencil_size;
        });

        // Projection
        samurai_new::for_each_cell_having_children<std::size_t>(mesh, [&] (std::size_t cell)
        {
            nnz[cell] = cfg::proj_stencil_size;
        });

        // Prediction
        samurai_new::for_each_cell_having_parent<std::size_t>(mesh, [&] (std::size_t cell)
        {
            nnz[cell] = cfg::pred_stencil_size;
        });

        return nnz;
    }


    template<class cfg, class Mesh, class GetCoefficientsFunc>
    void set_coefficients(Mat& A, const Mesh& mesh, const Stencil<cfg::scheme_stencil_size, Mesh::dim>& stencil, GetCoefficientsFunc &&get_coefficients)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;

        //--------------//
        //   Interior   //
        //--------------//

        StencilIndices<PetscInt, cfg::scheme_stencil_size, dim> stencil_indices(stencil);

        for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
        {
            auto coeffs = get_coefficients(h);

            for_each_stencil<PetscInt>(mesh, mesh[mesh_id_t::cells], level, stencil_indices,
            [&] (const std::array<PetscInt, cfg::scheme_stencil_size>& indices)
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

        });

        // Must flush to use ADD_VALUES instead of INSERT_VALUES
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);


        //--------------//
        //   Boundary   //
        //--------------//

        for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
        {
            // 1 - The boundary ghosts are 'eliminated' from the system, we simply add 1 on the diagonal.
            auto boundary_ghosts = difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.domain()).on(level);
            for_each_cell<PetscInt>(mesh, level, boundary_ghosts, [&](PetscInt ghost)
            {
                MatSetValue(A, ghost, ghost, 1, ADD_VALUES);
            });

            // 2 - The (opposite of the) contribution of the outer ghost is added to the diagonal of stencil center
            auto coeffs = get_coefficients(h);

            foreach_interval_on_boundary(mesh, level, stencil, coeffs,
            [&] (const auto& mesh_interval, const auto& towards_bdry_ghost, double out_coeff)
            {
                samurai_new::StencilIndices<PetscInt, 2, dim> in_out_indices(samurai_new::in_out_stencil<dim>(towards_bdry_ghost));
                samurai_new::for_each_stencil<PetscInt>(mesh, mesh_interval, in_out_indices, 
                [&] (const std::array<PetscInt, 2>& indices)
                {
                    auto& in_cell   = indices[0];
                    auto& out_ghost = indices[1];
                    MatSetValue(A, in_cell,   in_cell, -out_coeff, ADD_VALUES); // the coeff is added to the center of the stencil
                    MatSetValue(A, in_cell, out_ghost, -out_coeff, ADD_VALUES); // the coeff of the ghost is removed
                });
            });
        });


        // Must flush to use INSERT_VALUES instead of ADD_VALUES
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

    }


}} // end namespace