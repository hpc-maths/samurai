#pragma once
#include "fv/FV_scheme.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Useful sizes to define the sparsity pattern of the matrix and perform the preallocation.
         */
        template <PetscInt output_field_size_,
                  PetscInt neighbourhood_width_,
                  PetscInt scheme_stencil_size_,
                  PetscInt center_index_,
                  PetscInt contiguous_indices_start_     = 0,
                  PetscInt contiguous_indices_size_      = 0,
                  DirichletEnforcement dirichlet_enfcmt_ = Equation>
        struct CellBasedAssemblyConfig
        {
            static constexpr PetscInt output_field_size        = output_field_size_;
            static constexpr PetscInt neighbourhood_width      = neighbourhood_width_;
            static constexpr PetscInt scheme_stencil_size      = scheme_stencil_size_;
            static constexpr PetscInt center_index             = center_index_;
            static constexpr PetscInt contiguous_indices_start = contiguous_indices_start_;
            static constexpr PetscInt contiguous_indices_size  = contiguous_indices_size_;
        };

        template <std::size_t dim, std::size_t output_field_size, std::size_t neighbourhood_width = 1>
        using StarStencilFV = CellBasedAssemblyConfig<output_field_size,
                                                      neighbourhood_width,
                                                      // ---- Stencil size
                                                      // Cell-centered Finite Volume scheme:
                                                      // center + 'neighbourhood_width' neighbours in each Cartesian direction (2*dim
                                                      // directions) --> 1+2=3 in 1D
                                                      //                 1+4=5 in 2D
                                                      1 + 2 * dim * neighbourhood_width,
                                                      // ---- Index of the stencil center
                                                      // (as defined in star_stencil())
                                                      neighbourhood_width,
                                                      // ---- Start index and size of contiguous cell indices
                                                      // (as defined in star_stencil())
                                                      0,
                                                      1 + 2 * neighbourhood_width>;

        template <std::size_t output_field_size>
        using OneCellStencilFV = CellBasedAssemblyConfig<output_field_size,
                                                         // ---- Neighbourhood width
                                                         0,
                                                         // ---- Stencil size (only one cell)
                                                         1,
                                                         // ---- Index of the stencil center (as defined in center_only_stencil())
                                                         0,
                                                         // ---- Start index and size of contiguous cell indices
                                                         0,
                                                         0>;

        template <std::size_t output_field_size>
        using EmptyStencilFV = CellBasedAssemblyConfig<output_field_size,
                                                       // ---- Neighbourhood width
                                                       0,
                                                       // ---- Stencil size
                                                       0,
                                                       // ---- Index of the stencil center
                                                       0,
                                                       // ---- Start index and size of contiguous cell indices
                                                       0,
                                                       0>;

        template <class cfg, class bdry_cfg, class Field>
        class CellBasedScheme : public FVScheme<Field, cfg::output_field_size, bdry_cfg>
        {
            template <class Scheme1, class Scheme2>
            friend class FluxBasedScheme_Sum_CellBasedScheme;

          protected:

            using base_class = FVScheme<Field, cfg::output_field_size, bdry_cfg>;
            using base_class::cell_coeff;
            using base_class::col_index;
            using base_class::dim;
            using base_class::field_size;
            using base_class::m_is_row_empty;
            using base_class::m_mesh;
            using base_class::row_index;
            using base_class::set_current_insert_mode;

          public:

            using cfg_t                                    = cfg;
            using field_t                                  = Field;
            using field_value_type                         = typename Field::value_type; // double
            static constexpr std::size_t output_field_size = cfg::output_field_size;
            using local_matrix_t                           = typename detail::LocalMatrix<field_value_type,
                                                                output_field_size,
                                                                field_size>::Type; // 'double' if field_size = 1, 'xtensor' representing a
                                                                                                             // matrix otherwise

            using stencil_t         = Stencil<cfg::scheme_stencil_size, dim>;
            using get_coeffs_func_t = std::function<std::array<local_matrix_t, cfg::scheme_stencil_size>(double)>;

          protected:

            stencil_t m_stencil;
            get_coeffs_func_t m_get_coefficients;

          public:

            CellBasedScheme(Field& unknown, stencil_t s, get_coeffs_func_t get_coeffs)
                : base_class(unknown)
                , m_stencil(s)
                , m_get_coefficients(get_coeffs)
            {
            }

            auto& stencil() const
            {
                return m_stencil;
            }

          protected:

            // Data index in the given stencil
            inline auto local_col_index(unsigned int cell_local_index, unsigned int field_j) const
            {
                if constexpr (field_size == 1)
                {
                    return cell_local_index;
                }
                else if constexpr (Field::is_soa)
                {
                    return field_j * cfg::scheme_stencil_size + cell_local_index;
                }
                else
                {
                    return cell_local_index * field_size + field_j;
                }
            }

            inline auto local_row_index(unsigned int cell_local_index, unsigned int field_i) const
            {
                if constexpr (output_field_size == 1)
                {
                    return cell_local_index;
                }
                else if constexpr (Field::is_soa)
                {
                    return field_i * cfg::scheme_stencil_size + cell_local_index;
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
                if constexpr (cfg::scheme_stencil_size == 0)
                {
                    return;
                }

                auto coeffs = m_get_coefficients(cell_length(0));
                for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                {
                    PetscInt scheme_nnz_i = cfg::scheme_stencil_size * field_size;
                    if constexpr (Field::is_soa)
                    {
                        scheme_nnz_i = 0;
                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                        {
                            if constexpr (cfg::contiguous_indices_start > 0)
                            {
                                for (unsigned int c = 0; c < cfg::contiguous_indices_start; ++c)
                                {
                                    double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                    if (coeff != 0)
                                    {
                                        scheme_nnz_i++;
                                    }
                                }
                            }
                            if constexpr (cfg::contiguous_indices_size > 0)
                            {
                                for (unsigned int c = 0; c < cfg::contiguous_indices_size; ++c)
                                {
                                    double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                    if (coeff != 0)
                                    {
                                        scheme_nnz_i += cfg::contiguous_indices_size;
                                        break;
                                    }
                                }
                            }
                            if constexpr (cfg::contiguous_indices_start + cfg::contiguous_indices_size < cfg::scheme_stencil_size)
                            {
                                for (unsigned int c = cfg::contiguous_indices_start + cfg::contiguous_indices_size;
                                     c < cfg::scheme_stencil_size;
                                     ++c)
                                {
                                    double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                    if (coeff != 0)
                                    {
                                        scheme_nnz_i++;
                                    }
                                }
                            }
                        }
                    }
                    for_each_cell(m_mesh,
                                  [&](auto& cell)
                                  {
                                      nnz[this->row_index(cell, field_i)] = scheme_nnz_i;
                                  });
                }
            }

          protected:

            //-------------------------------------------------------------//
            //             Assemble scheme in the interior                 //
            //-------------------------------------------------------------//

            void assemble_scheme(Mat& A) override
            {
                if constexpr (cfg::scheme_stencil_size == 0)
                {
                    return;
                }

                set_current_insert_mode(INSERT_VALUES);

                // Apply the given coefficents to the given stencil
                for_each_stencil(
                    m_mesh,
                    m_stencil,
                    m_get_coefficients,
                    [&](const auto& cells, const auto& coeffs)
                    {
                        // std::cout << "coeffs: " << std::endl;
                        // for (std::size_t i=0; i<cfg::scheme_stencil_size; i++)
                        //     std::cout << i << ": " << coeffs[i] << std::endl;

                        // Global rows and columns
                        std::array<PetscInt, cfg::scheme_stencil_size * output_field_size> rows;
                        for (unsigned int c = 0; c < cfg::scheme_stencil_size; ++c)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                rows[local_row_index(c, field_i)] = static_cast<PetscInt>(row_index(cells[c], field_i));
                            }
                        }
                        std::array<PetscInt, cfg::scheme_stencil_size * field_size> cols;
                        for (unsigned int c = 0; c < cfg::scheme_stencil_size; ++c)
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
                        if constexpr (field_size == 1 || Field::is_soa)
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
                                auto stencil_center_row = static_cast<PetscInt>(row_index(cells[cfg::center_index], field_i));
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    if constexpr (cfg::contiguous_indices_start > 0)
                                    {
                                        for (unsigned int c = 0; c < cfg::contiguous_indices_start; ++c)
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
                                            if (coeff != 0)
                                            {
                                                MatSetValue(A, stencil_center_row, cols[local_col_index(c, field_j)], coeff, INSERT_VALUES);
                                            }
                                        }
                                    }
                                    if constexpr (cfg::contiguous_indices_size > 0)
                                    {
                                        std::array<double, cfg::contiguous_indices_size> contiguous_coeffs;
                                        for (unsigned int c = 0; c < cfg::contiguous_indices_size; ++c)
                                        {
                                            if constexpr (field_size == 1 && output_field_size == 1)
                                            {
                                                contiguous_coeffs[c] = coeffs[cfg::contiguous_indices_start + c];
                                            }
                                            else
                                            {
                                                contiguous_coeffs[c] = coeffs[cfg::contiguous_indices_start + c](field_i, field_j);
                                            }
                                        }
                                        if (std::any_of(contiguous_coeffs.begin(),
                                                        contiguous_coeffs.end(),
                                                        [](auto coeff)
                                                        {
                                                            return coeff != 0;
                                                        }))
                                        {
                                            MatSetValues(A,
                                                         1,
                                                         &stencil_center_row,
                                                         static_cast<PetscInt>(cfg::contiguous_indices_size),
                                                         &cols[local_col_index(cfg::contiguous_indices_start, field_j)],
                                                         contiguous_coeffs.data(),
                                                         INSERT_VALUES);
                                        }
                                    }
                                    if constexpr (cfg::contiguous_indices_start + cfg::contiguous_indices_size < cfg::scheme_stencil_size)
                                    {
                                        for (unsigned int c = cfg::contiguous_indices_start + cfg::contiguous_indices_size;
                                             c < cfg::scheme_stencil_size;
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
                                            if (coeff != 0)
                                            {
                                                MatSetValue(A, stencil_center_row, cols[local_col_index(c, field_j)], coeff, INSERT_VALUES);
                                            }
                                        }
                                    }

                                    m_is_row_empty[static_cast<std::size_t>(stencil_center_row)] = false;
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

                            for (unsigned int c = 0; c < cfg::scheme_stencil_size; ++c)
                            {
                                // Insert a coefficient block of size <output_field_size x field_size>:
                                // - in 'rows', for each cell, <output_field_size> rows are contiguous.
                                // - in 'cols', for each cell, <field_size> cols are contiguous.
                                // - coeffs[c] is a row-major matrix (xtensor), as requested by PETSc.
                                MatSetValues(A,
                                             static_cast<PetscInt>(output_field_size),
                                             &rows[local_row_index(cfg::center_index, 0)],
                                             static_cast<PetscInt>(field_size),
                                             &cols[local_col_index(c, 0)],
                                             coeffs[c].data(),
                                             INSERT_VALUES);
                            }

                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                auto row            = static_cast<std::size_t>(rows[local_row_index(cfg::center_index, field_i)]);
                                m_is_row_empty[row] = false;
                            }
                        }
                    });
            }
        };

        template <typename, typename = void>
        constexpr bool is_CellBasedScheme{};

        template <typename T>
        constexpr bool is_CellBasedScheme<T, std::void_t<decltype(std::declval<T>().stencil())>> = true;

    } // end namespace petsc
} // end namespace samurai
