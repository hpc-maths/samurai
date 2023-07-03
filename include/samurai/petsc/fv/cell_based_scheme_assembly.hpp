#pragma once
#include "../../schemes/fv/cell_based_scheme.hpp"
#include "FV_scheme_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <class Scheme>
        class Assembly<Scheme, std::enable_if_t<is_CellBasedScheme_v<Scheme>>> : public FVSchemeAssembly<Scheme>
        {
          protected:

            using base_class = FVSchemeAssembly<Scheme>;
            using base_class::col_index;
            using base_class::dim;
            using base_class::field_size;
            using base_class::row_index;
            using base_class::set_is_row_not_empty;

          public:

            using base_class::mesh;
            using base_class::scheme;
            using base_class::set_current_insert_mode;

          public:

            using scheme_t                                   = Scheme;
            using cfg_t                                      = typename Scheme::cfg_t;
            using bdry_cfg_t                                 = typename Scheme::bdry_cfg;
            using field_t                                    = typename Scheme::field_t;
            using field_value_type                           = typename field_t::value_type; // double
            static constexpr std::size_t output_field_size   = cfg_t::output_field_size;
            static constexpr std::size_t scheme_stencil_size = cfg_t::scheme_stencil_size;
            using local_matrix_t                             = typename detail::LocalMatrix<field_value_type,
                                                                output_field_size,
                                                                field_size>::Type; // 'double' if field_size = 1, 'xtensor' representing a
                                                                                                               // matrix otherwise

            using stencil_t         = Stencil<scheme_stencil_size, dim>;
            using get_coeffs_func_t = std::function<std::array<local_matrix_t, scheme_stencil_size>(double)>;

            explicit Assembly(const Scheme& scheme)
                : base_class(scheme)
            {
            }

          protected:

            // Data index in the given stencil
            inline auto local_col_index(unsigned int cell_local_index, [[maybe_unused]] unsigned int field_j) const
            {
                if constexpr (field_size == 1)
                {
                    return cell_local_index;
                }
                else if constexpr (field_t::is_soa)
                {
                    return field_j * scheme_stencil_size + cell_local_index;
                }
                else
                {
                    return cell_local_index * field_size + field_j;
                }
            }

            inline auto local_row_index(unsigned int cell_local_index, [[maybe_unused]] unsigned int field_i) const
            {
                if constexpr (output_field_size == 1)
                {
                    return cell_local_index;
                }
                else if constexpr (field_t::is_soa)
                {
                    return field_i * scheme_stencil_size + cell_local_index;
                }
                else
                {
                    return cell_local_index * output_field_size + field_i;
                }
            }

          public:

            //-------------------------------------------------------------//
            //                     Sparsity pattern                        //
            //-------------------------------------------------------------//

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                if constexpr (scheme_stencil_size == 0)
                {
                    return;
                }

                auto coeffs = scheme().coefficients(cell_length(0));
                for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                {
                    PetscInt scheme_nnz_i = scheme_stencil_size * field_size;
                    if constexpr (field_t::is_soa)
                    {
                        scheme_nnz_i = 0;
                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                        {
                            if constexpr (cfg_t::contiguous_indices_start > 0)
                            {
                                for (unsigned int c = 0; c < cfg_t::contiguous_indices_start; ++c)
                                {
                                    double coeff = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                    if (coeff != 0)
                                    {
                                        scheme_nnz_i++;
                                    }
                                }
                            }
                            if constexpr (cfg_t::contiguous_indices_size > 0)
                            {
                                for (unsigned int c = 0; c < cfg_t::contiguous_indices_size; ++c)
                                {
                                    double coeff = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                    if (coeff != 0)
                                    {
                                        scheme_nnz_i += cfg_t::contiguous_indices_size;
                                        break;
                                    }
                                }
                            }
                            if constexpr (cfg_t::contiguous_indices_start + cfg_t::contiguous_indices_size < cfg_t::scheme_stencil_size)
                            {
                                for (unsigned int c = cfg_t::contiguous_indices_start + cfg_t::contiguous_indices_size;
                                     c < scheme_stencil_size;
                                     ++c)
                                {
                                    double coeff = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                    if (coeff != 0)
                                    {
                                        scheme_nnz_i++;
                                    }
                                }
                            }
                        }
                    }
                    for_each_cell(mesh(),
                                  [&](auto& cell)
                                  {
                                      nnz[static_cast<std::size_t>(this->row_index(cell, field_i))] += scheme_nnz_i;
                                  });
                }
            }

          protected:

            template <class Func>
            void for_each_stencil_and_coeffs(Func&& f)
            {
                auto stencil    = scheme().stencil();
                auto stencil_it = make_stencil_iterator(mesh(), stencil);

                for_each_level(mesh(),
                               [&](std::size_t level)
                               {
                                   auto coeffs = scheme().coefficients(cell_length(level));

                                   for_each_stencil(mesh(),
                                                    level,
                                                    stencil_it,
                                                    [&](auto& cells)
                                                    {
                                                        f(cells, coeffs);
                                                    });
                               });
            }

            //-------------------------------------------------------------//
            //             Assemble scheme in the interior                 //
            //-------------------------------------------------------------//

          public:

            void assemble_scheme(Mat& A) override
            {
                // std::cout << "assemble_scheme() of " << this->name() << std::endl;

                if (this->current_insert_mode() == ADD_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(INSERT_VALUES);
                }

                // Apply the given coefficents to the given stencil
                for_each_stencil_and_coeffs(
                    [&](const auto& cells, const auto& coeffs)
                    {
                        // std::cout << "coeffs: " << std::endl;
                        // for (std::size_t i=0; i<scheme_stencil_size; i++)
                        //     std::cout << i << ": " << coeffs[i] << std::endl;

                        // Global rows and columns
                        std::array<PetscInt, cfg_t::scheme_stencil_size * output_field_size> rows;
                        for (unsigned int c = 0; c < cfg_t::scheme_stencil_size; ++c)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                rows[local_row_index(c, field_i)] = static_cast<PetscInt>(row_index(cells[c], field_i));
                            }
                        }
                        std::array<PetscInt, cfg_t::scheme_stencil_size * field_size> cols;
                        for (unsigned int c = 0; c < cfg_t::scheme_stencil_size; ++c)
                        {
                            for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                            {
                                cols[local_col_index(c, field_j)] = static_cast<PetscInt>(col_index(cells[c], field_j));
                            }
                        }

                        // The stencil coefficients are stored as an array of
                        // matrices. For instance, vector diffusion in 2D:
                        //
                        //                        L     R     C     B     T   (left, right, center, bottom, top)
                        //     field_i (Lap_x) |-1   |-1   | 4   |-1   |-1   |
                        //     field_j (Lap_y) |   -1|   -1|    4|   -1|   -1|
                        //
                        // Other example, gradient in 2D:
                        //
                        //                        L  R  C  B  T
                        //     field_i (Grad_x) |-1| 1|  |  |  |
                        //     field_j (Grad_y) |  |  |  |-1| 1|

                        // Coefficient insertion
                        if constexpr (field_size == 1 || field_t::is_soa)
                        {
                            // In SOA, the indices are ordered in field_i for
                            // all cells, then field_j for all cells:
                            //
                            // - Diffusion example:
                            //            [         field_i        |         field_j        ]
                            //            [  L    R    C    B    T |  L    R    C    B    T ]
                            //  coupling: [ i j| i j| i j| i j| i j| i j| i j| i j| i j| i j]
                            //            [-1 0|-1 0| 4 0|-1 0|-1 0|0 -1|0 -1|0
                            //            4|0 -1|0 -1]
                            //
                            // For the cell of global index c:
                            //
                            //                field_i       ...       field_j
                            //   row c*i: |-1 -1  4 -1 -1|  ...  | 0  0  0  0 0|
                            //
                            //   row c*j: | 0  0  0  0  0|  ...  |-1 -1  4 -1
                            //   -1|
                            //                |_______|              |_______|
                            //               contiguous              contiguous
                            //
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                auto stencil_center_row = static_cast<PetscInt>(row_index(cells[cfg_t::center_index], field_i));
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    if constexpr (cfg_t::contiguous_indices_start > 0)
                                    {
                                        for (unsigned int c = 0; c < cfg_t::contiguous_indices_start; ++c)
                                        {
                                            double coeff;
                                            if constexpr (field_size == 1 && output_field_size == 1)
                                            {
                                                coeff = coeffs[c];
                                            }
                                            else
                                            {
                                                coeff = coeffs[c](field_i, field_j);
                                            }
                                            if (coeff != 0 || stencil_center_row == cols[local_col_index(c, field_j)])
                                            {
                                                MatSetValue(A, stencil_center_row, cols[local_col_index(c, field_j)], coeff, INSERT_VALUES);
                                            }
                                        }
                                    }
                                    if constexpr (cfg_t::contiguous_indices_size > 0)
                                    {
                                        std::array<double, cfg_t::contiguous_indices_size> contiguous_coeffs;
                                        for (unsigned int c = 0; c < cfg_t::contiguous_indices_size; ++c)
                                        {
                                            if constexpr (field_size == 1 && output_field_size == 1)
                                            {
                                                contiguous_coeffs[c] = coeffs[cfg_t::contiguous_indices_start + c];
                                            }
                                            else
                                            {
                                                contiguous_coeffs[c] = coeffs[cfg_t::contiguous_indices_start + c](field_i, field_j);
                                            }
                                        }
                                        // if (std::any_of(contiguous_coeffs.begin(),
                                        //                 contiguous_coeffs.end(),
                                        //                 [](auto coeff)
                                        //                 {
                                        //                     return coeff != 0;
                                        //                 }))
                                        // {
                                        MatSetValues(A,
                                                     1,
                                                     &stencil_center_row,
                                                     static_cast<PetscInt>(cfg_t::contiguous_indices_size),
                                                     &cols[local_col_index(cfg_t::contiguous_indices_start, field_j)],
                                                     contiguous_coeffs.data(),
                                                     INSERT_VALUES);
                                        // }
                                    }
                                    if constexpr (cfg_t::contiguous_indices_start + cfg_t::contiguous_indices_size
                                                  < cfg_t::scheme_stencil_size)
                                    {
                                        for (unsigned int c = cfg_t::contiguous_indices_start + cfg_t::contiguous_indices_size;
                                             c < cfg_t::scheme_stencil_size;
                                             ++c)
                                        {
                                            double coeff;
                                            if constexpr (field_size == 1 && output_field_size == 1)
                                            {
                                                coeff = coeffs[c];
                                            }
                                            else
                                            {
                                                coeff = coeffs[c](field_i, field_j);
                                            }
                                            if (coeff != 0 || stencil_center_row == cols[local_col_index(c, field_j)])
                                            {
                                                MatSetValue(A, stencil_center_row, cols[local_col_index(c, field_j)], coeff, INSERT_VALUES);
                                            }
                                        }
                                    }

                                    set_is_row_not_empty(stencil_center_row);
                                }
                            }
                        }
                        else // AOS
                        {
                            // In AOS, the blocks of coefficients are inserted
                            // as given by the user:
                            //
                            //                     i  j  i  j  i  j  i  j  i  j
                            // row (c*2)+i   --> [-1  0|-1  0| 4  0|-1  0|-1  0]
                            // row (c*2)+i+1 --> [ 0 -1| 0 -1| 0  4| 0 -1| 0 -1]

                            for (unsigned int c = 0; c < scheme_stencil_size; ++c)
                            {
                                // Insert a coefficient block of size <output_field_size x field_size>:
                                // - in 'rows', for each cell, <output_field_size> rows are contiguous.
                                // - in 'cols', for each cell, <field_size> cols are contiguous.
                                // - coeffs[c] is a row-major matrix (xtensor), as requested by PETSc.
                                MatSetValues(A,
                                             static_cast<PetscInt>(output_field_size),
                                             &rows[local_row_index(cfg_t::center_index, 0)],
                                             static_cast<PetscInt>(field_size),
                                             &cols[local_col_index(c, 0)],
                                             coeffs[c].data(),
                                             INSERT_VALUES);
                            }

                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                auto row = rows[local_row_index(cfg_t::center_index, field_i)];
                                set_is_row_not_empty(row);
                            }
                        }
                    });
            }
        };

    } // end namespace petsc
} // end namespace samurai
