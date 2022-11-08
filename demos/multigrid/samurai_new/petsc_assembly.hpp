#pragma once
#include <petsc.h>
#include <samurai/algorithm.hpp>
#include "indices.hpp"

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
        std::size_t n = mesh.nb_cells();

        // Number of non-zeros per row. By default, the stencil size.
        std::vector<PetscInt> nnz(n, cfg::scheme_stencil_size);

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


    template<class cfg, 
             class Mesh,
             class GetCoefficientsFunc>
    void set_coefficients(Mat& A, const Mesh& mesh, const StencilShape<Mesh::dim, cfg::scheme_stencil_size>& stencil_shape, GetCoefficientsFunc &&get_coefficients)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;

        StencilIndices<PetscInt, dim, cfg::scheme_stencil_size> stencil(stencil_shape);

        for_each_level(mesh[mesh_id_t::cells], [&](std::size_t level, double h)
        {
            std::array<double, cfg::scheme_stencil_size> coeffs = get_coefficients(h);

            for_each_stencil<PetscInt>(mesh, mesh[mesh_id_t::cells], level, stencil,
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
    }


}} // end namespace