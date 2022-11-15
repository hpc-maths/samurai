#pragma once
#include <petsc.h>
#include <samurai/algorithm.hpp>
#include "indices.hpp"
#include "boundary.hpp"

namespace samurai_new { namespace petsc
{
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
    void set_coefficients(Mat& A, const Mesh& mesh, const StencilShape<Mesh::dim, cfg::scheme_stencil_size>& stencil, GetCoefficientsFunc &&get_coefficients)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;

        //--------------//
        //   Interior   //
        //--------------//

        StencilIndices<PetscInt, dim, cfg::scheme_stencil_size> stencil_indices(stencil);

        for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
        {
            std::array<double, cfg::scheme_stencil_size> coeffs = get_coefficients(h);

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
            auto boundary_ghosts = difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.domain());
            for_each_cell<PetscInt>(mesh, level, boundary_ghosts, [&](PetscInt ghost)
            {
                MatSetValue(A, ghost, ghost, 1, ADD_VALUES);
            });

            // 2 - The (opposite of the) contribution of the outer ghost is added to the diagonal of stencil center
            auto coeffs = get_coefficients(h);

            in_boundary(mesh, level, stencil,
            [&] (const auto& mesh_interval, const auto& towards_bdry_ghost)
            {
                // The vector towards_bdry_ghost is searched in the stencil to identify the coefficient associated to the ghost
                unsigned int out_coeff_index = 0;
                for (unsigned int is = 0; is<cfg::scheme_stencil_size; ++is)
                {
                    auto direction = xt::view(stencil, is);
                    if (xt::all(xt::eval(xt::equal(direction, towards_bdry_ghost)))) //if (direction == towards_bdry_ghost)
                    {
                        out_coeff_index = is;
                        break;
                    }
                }
                double out_coeff = coeffs[out_coeff_index];

                // The contribution is added to the center of the stencil
                samurai_new::for_each_cell<PetscInt>(mesh, mesh_interval, [&] (PetscInt cell)
                {
                    MatSetValue(A, cell, cell , -out_coeff, ADD_VALUES); 
                });
            });
        });

    }


}} // end namespace