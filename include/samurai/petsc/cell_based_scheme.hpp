#pragma once
#include "matrix_assembly.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "../boundary.hpp"

namespace samurai 
{ 
    namespace petsc
    {
        namespace detail
        {
            /**
             * Local square matrix to store the coefficients of a vectorial field.
            */
            template<class value_type, std::size_t rows, std::size_t cols=rows>
            struct LocalMatrix
            {
                using Type = xt::xtensor_fixed<value_type, xt::xshape<rows, cols>>;
            };

            /**
             * Template specialization: if size=1, then just a scalar coefficient
            */
            template<class value_type>
            struct LocalMatrix<value_type, 1, 1>
            {
                using Type = value_type;
            };
        }

        template<class matrix_type>
        matrix_type eye()
        {
            static constexpr auto s = typename matrix_type::shape_type();
            return xt::eye(s[0]);
        }

        template<>
        double eye<double>()
        {
            return 1;
        }

        template<class matrix_type>
        matrix_type zeros()
        {
            static constexpr auto s = typename matrix_type::shape_type();
            return xt::zeros(s[0], s[1]);
        }

        template<>
        double zeros<double>()
        {
            return 0;
        }




        

        template<class cfg, class Field>
        class CellBasedScheme : public MatrixAssembly
        {
        public:
            using cfg_t = cfg;
            using field_t = Field;

            using Mesh = typename Field::mesh_t;
            using field_value_type = typename Field::value_type; // double
            static constexpr std::size_t field_size = Field::size;
            static constexpr std::size_t output_field_size = cfg::output_field_size;
            using local_matrix_t = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type; // 'double' if field_size = 1, 'xtensor' representing a matrix otherwise
            using mesh_id_t = typename Mesh::mesh_id_t;
            static constexpr std::size_t dim = Mesh::dim;

            using stencil_t = Stencil<cfg::scheme_stencil_size, dim>;
            using GetCoefficientsFunc = std::function<std::array<local_matrix_t, cfg::scheme_stencil_size>(double)>;
            using boundary_condition_t = typename Field::boundary_condition_t;
        
            using MatrixAssembly::assemble_matrix;
        protected:
            Field& m_unknown;
            Mesh& m_mesh;
            std::size_t m_n_cells;
            stencil_t m_stencil;
            GetCoefficientsFunc m_get_coefficients;
            const std::vector<boundary_condition_t>& m_boundary_conditions;
            std::vector<bool> m_is_row_empty;
        public:
            CellBasedScheme(Field& unknown, stencil_t s, GetCoefficientsFunc get_coeffs) :
                m_unknown(unknown), 
                m_mesh(unknown.mesh()), 
                m_stencil(s), 
                m_get_coefficients(get_coeffs), 
                m_boundary_conditions(unknown.boundary_conditions())
            {
                m_n_cells = m_mesh.nb_cells();
                m_is_row_empty = std::vector(static_cast<std::size_t>(matrix_rows()), true);
            }

            auto& unknown() const
            {
                return m_unknown;
            }

            auto& mesh() const
            {
                return m_mesh;
            }

            auto& stencil() const
            {
                return m_stencil;
            }

            const auto& boundary_conditions() const
            {
                return m_boundary_conditions;
            }

            PetscInt matrix_rows() const override
            {
                return static_cast<PetscInt>(m_n_cells * output_field_size);
            }

            PetscInt matrix_cols() const override
            {
                return static_cast<PetscInt>(m_n_cells * field_size);
            }

        private:
            // Global data index
            inline PetscInt col_index(PetscInt cell_index, unsigned int field_j) const
            {
                if constexpr (field_size == 1)
                {
                    return cell_index;
                }
                else if constexpr (Field::is_soa)
                {
                    return static_cast<PetscInt>(field_j * m_n_cells) + cell_index;
                }
                else
                {
                    return cell_index * static_cast<PetscInt>(field_size) + static_cast<PetscInt>(field_j);
                }
            }
            inline PetscInt row_index(PetscInt cell_index, unsigned int field_i) const
            {
                if constexpr (output_field_size == 1)
                {
                    return cell_index;
                }
                else if constexpr (Field::is_soa)
                {
                    return static_cast<PetscInt>(field_i * m_n_cells) + cell_index;
                }
                else
                {
                    return cell_index * static_cast<PetscInt>(output_field_size) + static_cast<PetscInt>(field_i);
                }
            }
            template <class CellT>
            inline auto col_index(const CellT& cell, unsigned int field_j) const
            {
                if constexpr (field_size == 1)
                {
                    return cell.index;
                }
                else if constexpr (Field::is_soa)
                {
                    return field_j * m_n_cells + cell.index;
                }
                else
                {
                    return cell.index * field_size + field_j;
                }
            }
            template <class CellT>
            inline auto row_index(const CellT& cell, unsigned int field_i) const
            {
                if constexpr (output_field_size == 1)
                {
                    return cell.index;
                }
                else if constexpr (Field::is_soa)
                {
                    return field_i * m_n_cells + cell.index;
                }
                else
                {
                    return cell.index * output_field_size + field_i;
                }
            }

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
            std::vector<PetscInt> sparsity_pattern() const override
            {
                // Number of non-zeros per row. 
                // 1 by default (for the unused ghosts outside of the domain).
                std::vector<PetscInt> nnz(static_cast<std::size_t>(matrix_rows()), 1);


                // Cells
                auto coeffs = m_get_coefficients(cell_length(0));
                for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                {
                    PetscInt scheme_nnz_i = cfg::scheme_stencil_size * field_size;
                    if constexpr (Field::is_soa)
                    {
                        scheme_nnz_i = 0;
                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                        {
                            if constexpr(cfg::contiguous_indices_start > 0)
                            {
                                for (unsigned int c=0; c<cfg::contiguous_indices_start; ++c)
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
                                        scheme_nnz_i++;
                                    }
                                }
                            }
                            if constexpr(cfg::contiguous_indices_size > 0)
                            {
                                for (unsigned int c=0; c<cfg::contiguous_indices_size; ++c)
                                {
                                    double coeff;
                                    if constexpr (field_size == 1 && output_field_size == 1)
                                    {
                                        coeff = coeffs[cfg::contiguous_indices_start + c];
                                    }
                                    else
                                    {
                                        coeff = coeffs[cfg::contiguous_indices_start + c](field_i, field_j);
                                    }
                                    if (coeff != 0)
                                    {
                                        scheme_nnz_i += cfg::contiguous_indices_size;
                                        break;
                                    }
                                }
                            }
                            if constexpr(cfg::contiguous_indices_start + cfg::contiguous_indices_size < cfg::scheme_stencil_size)
                            {
                                for (unsigned int c=cfg::contiguous_indices_start + cfg::contiguous_indices_size; c<cfg::scheme_stencil_size; ++c)
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
                                        scheme_nnz_i++;
                                    }
                                }
                            }

                        }
                    }
                    for_each_cell(m_mesh, [&](auto& cell)
                    {
                        nnz[row_index(cell, field_i)] = scheme_nnz_i;
                    });
                }

                // Boundary conditions
                if (this->include_bc())
                {
                    sparsity_pattern_boundary(nnz);
                }

                // Projection
                for_each_projection_ghost(m_mesh, [&](auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        nnz[row_index(ghost, field_i)] = cfg::proj_stencil_size;
                    }
                });

                // Prediction
                for_each_prediction_ghost(m_mesh, [&](auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        nnz[row_index(ghost, field_i)] = cfg::pred_stencil_size;
                    }
                });

                return nnz;
            }

        protected:
            virtual void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const
            {
                // Boundary ghosts on the Dirichlet boundary (if Elimination, nnz=1, the default value)
                if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Equation)
                {
                    for_each_stencil_center_and_outside_ghost(m_mesh, m_stencil, [&](const auto& cells, const auto& towards_ghost)
                    {
                        auto& cell  = cells[0];
                        auto& ghost = cells[1];
                        auto boundary_point = cell.face_center(towards_ghost);
                        auto bc = find(m_boundary_conditions, boundary_point);
                        if (bc.is_dirichlet())
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                nnz[row_index(ghost, field_i)] = 2;
                            }
                        }
                    });
                }

                // Boundary ghosts on the Neumann boundary
                if (has_neumann(m_boundary_conditions))
                {
                    for_each_stencil_center_and_outside_ghost(m_mesh, m_stencil, [&](const auto& cells, const auto& towards_ghost)
                    {
                        auto& cell  = cells[0];
                        auto& ghost = cells[1];
                        auto boundary_point = cell.face_center(towards_ghost);
                        auto bc = find(m_boundary_conditions, boundary_point);
                        if (bc.is_neumann())
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                nnz[row_index(ghost, field_i)] = 2;
                            }
                        }
                    });
                }
            }

        private:
            void assemble_scheme_on_uniform_grid(Mat& A) override
            {
                // Apply the given coefficents to the given stencil
                for_each_stencil(m_mesh, m_stencil, m_get_coefficients,
                [&] (const auto& cells, const auto& coeffs)
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

                    // The stencil coefficients are stored as an array of matrices.
                    // For instance, vector diffusion in 2D:
                    //
                    //                        L     R     C     B     T        (left, right, center, bottom, top)
                    //     field_i (Lap_x) |-1   |-1   | 4   |-1   |-1   |
                    //     field_j (Lap_y) |   -1|   -1|    4|   -1|   -1|
                    //
                    // Other example, gradient in 2D:
                    //
                    //                        L  R  C  B  T 
                    //     field_i (Grad_x) |-1| 1|  |  |  |
                    //     field_j (Grad_y) |  |  |  |-1| 1|
                    
                    // Coefficient insertion
                    if constexpr(field_size == 1 || Field::is_soa)
                    {
                        // In SOA, the indices are ordered in field_i for all cells, then field_j for all cells:
                        //
                        // - Diffusion example:
                        //            [         field_i        |         field_j        ]
                        //            [  L    R    C    B    T |  L    R    C    B    T ]
                        //  coupling: [ i j| i j| i j| i j| i j| i j| i j| i j| i j| i j]
                        //            [-1 0|-1 0| 4 0|-1 0|-1 0|0 -1|0 -1|0  4|0 -1|0 -1]
                        //
                        // For the cell of global index c:
                        //
                        //                field_i       ...       field_j 
                        //   row c*i: |-1 -1  4 -1 -1|  ...  | 0  0  0  0  0|
                        //
                        //   row c*j: | 0  0  0  0  0|  ...  |-1 -1  4 -1 -1|
                        //                |_______|              |_______|
                        //               contiguous              contiguous
                        //         
                        for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            auto stencil_center_row = static_cast<PetscInt>(row_index(cells[cfg::center_index], field_i));
                            for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                            {

                                if constexpr(cfg::contiguous_indices_start > 0)
                                {
                                    for (unsigned int c=0; c<cfg::contiguous_indices_start; ++c)
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
                                if constexpr(cfg::contiguous_indices_size > 0)
                                {
                                    std::array<double, cfg::contiguous_indices_size> contiguous_coeffs;
                                    for (unsigned int c=0; c<cfg::contiguous_indices_size; ++c)
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
                                    if (std::any_of(contiguous_coeffs.begin(), contiguous_coeffs.end(), [](auto coeff){ return coeff != 0; }))
                                    {
                                        MatSetValues(A, 1, &stencil_center_row, static_cast<PetscInt>(cfg::contiguous_indices_size), &cols[local_col_index(cfg::contiguous_indices_start, field_j)], contiguous_coeffs.data(), INSERT_VALUES);
                                    }
                                }
                                if constexpr(cfg::contiguous_indices_start + cfg::contiguous_indices_size < cfg::scheme_stencil_size)
                                {
                                    for (unsigned int c=cfg::contiguous_indices_start + cfg::contiguous_indices_size; c<cfg::scheme_stencil_size; ++c)
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
                        // In AOS, the blocks of coefficients are inserted as given by the user:
                        //
                        //                     i  j  i  j  i  j  i  j  i  j
                        // row (c*2)+i   --> [-1  0|-1  0| 4  0|-1  0|-1  0]
                        // row (c*2)+i+1 --> [ 0 -1| 0 -1| 0  4| 0 -1| 0 -1]

                        for (unsigned int c=0; c<cfg::scheme_stencil_size; ++c)
                        {
                            // Insert a coefficient block of size in <output_field_size x field_size>:
                            // - in 'rows', for each cell, <output_field_size> rows are contiguous.
                            // - in 'cols', for each cell,        <field_size> cols are contiguous.
                            // - coeffs[c] is a row-major matrix (xtensor), as requested by PETSc.
                            MatSetValues(A, static_cast<PetscInt>(output_field_size), &rows[local_row_index(cfg::center_index, 0)], static_cast<PetscInt>(field_size), &cols[local_col_index(c, 0)], coeffs[c].data(), INSERT_VALUES);
                        }

                        for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            auto row = static_cast<std::size_t>(rows[local_row_index(cfg::center_index, field_i)]);
                            m_is_row_empty[row] = false;
                        }
                    }
                });
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                {
                    // Must flush to use ADD_VALUES instead of INSERT_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                }

                for_each_stencil_center_and_outside_ghost(m_mesh, m_stencil, m_get_coefficients, 
                [&] (const auto& cells, const auto& towards_ghost, auto& ghost_coeff)
                {
                    const auto& cell  = cells[0];
                    const auto& ghost = cells[1];
                    auto boundary_point = cell.face_center(towards_ghost);
                    auto bc = find(m_boundary_conditions, boundary_point);

                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt cell_index = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));
                        double coeff;
                        if constexpr (field_size == 1 && output_field_size == 1)
                        {
                            coeff = ghost_coeff;
                        }
                        else
                        {
                            coeff = ghost_coeff(field_i, field_i);
                        }
                        if (bc.is_dirichlet())
                        {
                            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                            {
                                // We have (u_ghost + u_cell)/2 = dirichlet_value ==> u_ghost = 2*dirichlet_value - u_cell
                                // The equation on the cell row is
                                //                     coeff*u_ghost + coeff_cell*u_cell + ... = f
                                // Eliminating u_ghost, it gives
                                //                           (coeff_cell - coeff)*u_cell + ... = f - 2*coeff*dirichlet_value
                                // which means that:
                                // - on the cell row, we have to 1) remove the coeff in the column of the ghost, 
                                //                               2) substract coeff in the column of the cell.
                                // - on the cell row of the right-hand side, we have to add -2*coeff*dirichlet_value.
                                MatSetValue(A, cell_index, ghost_index, -coeff, ADD_VALUES); // the coeff of the ghost is removed from the stencil (we want 0 so we substract the coeff we set before)
                                MatSetValue(A, cell_index, cell_index,  -coeff, ADD_VALUES); // the coeff is substracted from the center of the stencil
                                MatSetValue(A, ghost_index, ghost_index,     1, ADD_VALUES); // 1 is added to the diagonal of the ghost
                            }
                            else
                            {
                                coeff = coeff == 0 ? 1 : coeff;
                                // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is [  1/2    1/2 ] = dirichlet_value
                                // which is equivalent to                                                         [-coeff -coeff] = -2 * coeff * dirichlet_value
                                MatSetValue(A, ghost_index, ghost_index, -coeff, INSERT_VALUES);
                                MatSetValue(A, ghost_index, cell_index , -coeff, INSERT_VALUES);
                            }
                        }
                        else
                        {
                            coeff = coeff == 0 ? 1 : coeff;
                            // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is [  1/h  -1/h ] = neumann_value             
                            // However, to have symmetry, we want to have coeff as the off-diagonal coefficient, so     [-coeff coeff] = -coeff * h * neumann_value
                            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                            {
                                MatSetValue(A, ghost_index, ghost_index, -coeff, ADD_VALUES); ////////// REMOVE THIS COMMENT// We want -coeff in the matrix, but we added 1 before, so we remove it
                                MatSetValue(A, ghost_index, cell_index,   coeff, ADD_VALUES);
                            }
                            else
                            {
                                MatSetValue(A, ghost_index, ghost_index, -coeff, INSERT_VALUES);
                                MatSetValue(A, ghost_index, cell_index,   coeff, INSERT_VALUES);
                            }
                        }

                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                });

                if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                }
            }

            void add_1_on_diag_for_useless_ghosts(Mat& A) override
            {
                /*for_each_outside_ghost(m_mesh, [&](const auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                    {
                        auto ghost_row = static_cast<PetscInt>(row_index(ghost, field_i));
                        if (m_is_row_empty[static_cast<std::size_t>(ghost_row)])
                        {
                        //for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                        //{
                        // auto ghost_col = static_cast<PetscInt>(row_index(ghost, field_j));
                            MatSetValue(A, ghost_row, ghost_row, 1, INSERT_VALUES);
                            m_is_row_empty[static_cast<std::size_t>(ghost_row)] = false;
                        //}
                        }
                    }
                });*/

                for (std::size_t i = 0; i<m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        MatSetValue(A, i, i, 1, INSERT_VALUES);
                        m_is_row_empty[i] = false;
                    }
                }
            }


        public:
            virtual void enforce_bc(Vec& b) const
            {
                for_each_stencil_center_and_outside_ghost(m_mesh, m_stencil, m_get_coefficients,
                [&] (const auto& cells, const auto& towards_ghost, auto& ghost_coeff)
                {
                    auto& cell  = cells[0];
                    auto& ghost = cells[1];
                    auto boundary_point = cell.face_center(towards_ghost);
                    auto bc = find(m_boundary_conditions, boundary_point);

                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt cell_index = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));
                        double coeff;
                        if constexpr (field_size == 1 && output_field_size == 1)
                        {
                            coeff = ghost_coeff;
                        }
                        else
                        {
                            coeff = ghost_coeff(field_i, field_i);
                        }
                        if (bc.is_dirichlet())
                        {
                            double dirichlet_value;
                            if constexpr (field_size == 1)
                            {
                                dirichlet_value = bc.get_value(boundary_point);
                            }
                            else
                            {
                                dirichlet_value = bc.get_value(boundary_point)(field_i); // TODO: call get_value() only once instead of once per field_i
                            }
                            
                            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                            {
                                //std::cout << "ADD " << (- 2 * coeff * dirichlet_value) << " to row " << cell_index << " (field " << field_i << ", cell " << cell.index << ", ghost " << ghost.index << ")" << std::endl;
                                VecSetValue(b, cell_index, - 2 * coeff * dirichlet_value, ADD_VALUES);
                            }
                            else
                            {
                                coeff = coeff == 0 ? 1 : coeff;
                                VecSetValue(b, ghost_index, - 2 * coeff * dirichlet_value, ADD_VALUES); // ADD_VALUES ?
                            }
                        }
                        else
                        {
                            coeff = coeff == 0 ? 1 : coeff;
                            auto& h = cell.length;
                            double neumann_value;
                            if constexpr (field_size == 1)
                            { 
                                neumann_value = bc.get_value(boundary_point);
                            }
                            else
                            {
                                neumann_value = bc.get_value(boundary_point)(field_i); // TODO: call get_value() only once instead of once per field_i
                            }
                            //std::cout << "ADD " << (- coeff * h * neumann_value) << " to row " << ghost_index << " (field " << field_i << ", cell " << cell.index << ", ghost " << ghost.index << ")" << std::endl;
                            VecSetValue(b, ghost_index, -coeff * h * neumann_value, ADD_VALUES); // ADD_VALUES ?
                        }
                    }
                });
            }

            virtual void enforce_projection_prediction(Vec& b) const
            {
                // Projection
                for_each_projection_ghost(m_mesh, [&](auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                    {
                        VecSetValue(b, static_cast<PetscInt>(col_index(ghost, field_i)), 0, INSERT_VALUES);
                    }
                });

                // Prediction
                for_each_prediction_ghost(m_mesh, [&](auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                    {
                        VecSetValue(b, static_cast<PetscInt>(col_index(ghost, field_i)), 0, INSERT_VALUES);
                    }
                });
            }


        private:
            void assemble_projection(Mat& A) override
            {
                static constexpr PetscInt number_of_children = (1 << dim);

                for_each_projection_ghost_and_children_cells<PetscInt>(m_mesh, 
                [&] (PetscInt ghost, const std::array<PetscInt, number_of_children>& children)
                {
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt ghost_index = row_index(ghost, field_i);
                        MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);
                        for (unsigned int i=0; i<number_of_children; ++i)
                        {
                            MatSetValue(A, ghost_index, col_index(children[i], field_i), -1./number_of_children, INSERT_VALUES);
                        }
                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                });
            }

            void assemble_prediction(Mat& A) override
            {
                static_assert(dim >= 1 && dim <= 3, "assemble_prediction() is not implemented for this dimension.");
                if constexpr (dim == 1)
                {
                    assemble_prediction_1D(A);
                }
                else if constexpr (dim == 2)
                {
                    assemble_prediction_2D(A);
                }
                else if constexpr (dim == 3)
                {
                    assemble_prediction_3D(A);
                }
            }

            void assemble_prediction_1D(Mat& A)
            {
                std::array<double, 3> pred{{1./8, 0, -1./8}};
                for_each_prediction_ghost(m_mesh, [&](auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt ghost_index = static_cast<PetscInt>(row_index(ghost, field_i));
                        MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                        auto ii = ghost.indices(0);
                        int sign_i = (ii & 1) ? -1 : 1;

                        auto parent_index = col_index(static_cast<PetscInt>(m_mesh.get_index(ghost.level - 1, ii/2)), field_i);
                        auto parent_left  = col_index(parent_index - 1, field_i);
                        auto parent_right = col_index(parent_index + 1, field_i);
                        MatSetValue(A, ghost_index, parent_index,                -1, INSERT_VALUES);
                        MatSetValue(A, ghost_index, parent_left,  -sign_i * pred[0], INSERT_VALUES);
                        MatSetValue(A, ghost_index, parent_right, -sign_i * pred[2], INSERT_VALUES);
                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                });

                /*using mesh_id_t = typename Mesh::mesh_id_t;

                auto min_level = mesh[mesh_id_t::cells].min_level();
                auto max_level = mesh[mesh_id_t::cells].max_level();
                for(std::size_t level=min_level+1; level<=max_level; ++level)
                {
                    auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                                    mesh[mesh_id_t::cells][level-1])
                            .on(level);

                    //std::array<double, 3> pred{{1./8, 0, -1./8}};
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
                }*/
            }

            void assemble_prediction_2D(Mat& A)
            {
                std::array<double, 3> pred{{1./8, 0, -1./8}};
                for_each_prediction_ghost(m_mesh, [&](auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                    {
                        PetscInt ghost_index = static_cast<PetscInt>(row_index(ghost, field_i));
                        MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                        auto ii = ghost.indices(0);
                        auto  j = ghost.indices(1);
                        int sign_i = (ii & 1) ? -1 : 1;
                        int sign_j =  (j & 1) ? -1 : 1;

                        auto parent              = col_index(static_cast<PetscInt>(m_mesh.get_index(ghost.level - 1, ii/2, j/2    )), field_i);
                        auto parent_bottom       = col_index(static_cast<PetscInt>(m_mesh.get_index(ghost.level - 1, ii/2, j/2 - 1)), field_i);
                        auto parent_top          = col_index(static_cast<PetscInt>(m_mesh.get_index(ghost.level - 1, ii/2, j/2 + 1)), field_i);
                        auto parent_left         = col_index(parent - 1       , field_i);
                        auto parent_right        = col_index(parent + 1       , field_i);
                        auto parent_bottom_left  = col_index(parent_bottom - 1, field_i);
                        auto parent_bottom_right = col_index(parent_bottom + 1, field_i);
                        auto parent_top_left     = col_index(parent_top - 1   , field_i);
                        auto parent_top_right    = col_index(parent_top + 1   , field_i);

                        MatSetValue(A, ghost_index, parent             ,                                  -1, INSERT_VALUES);
                        MatSetValue(A, ghost_index, parent_bottom      ,                   -sign_j * pred[0], INSERT_VALUES); //        sign_j * -1/8
                        MatSetValue(A, ghost_index, parent_top         ,                   -sign_j * pred[2], INSERT_VALUES); //        sign_j *  1/8
                        MatSetValue(A, ghost_index, parent_left        ,                   -sign_i * pred[0], INSERT_VALUES); // sign_i        * -1/8
                        MatSetValue(A, ghost_index, parent_right       ,                   -sign_i * pred[2], INSERT_VALUES); // sign_i        *  1/8
                        MatSetValue(A, ghost_index, parent_bottom_left , sign_i * sign_j * pred[0] * pred[0], INSERT_VALUES); // sign_i*sign_j *  1/64
                        MatSetValue(A, ghost_index, parent_bottom_right, sign_i * sign_j * pred[2] * pred[0], INSERT_VALUES); // sign_i*sign_j * -1/64
                        MatSetValue(A, ghost_index, parent_top_left    , sign_i * sign_j * pred[0] * pred[2], INSERT_VALUES); // sign_i*sign_j * -1/64
                        MatSetValue(A, ghost_index, parent_top_right   , sign_i * sign_j * pred[2] * pred[2], INSERT_VALUES); // sign_i*sign_j *  1/64
                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                });

                /*using mesh_id_t = typename Mesh::mesh_id_t;

                auto min_level = m_mesh[mesh_id_t::cells].min_level();
                auto max_level = m_mesh[mesh_id_t::cells].max_level();
                for(std::size_t level=min_level+1; level<=max_level; ++level)
                {
                    auto set = intersection(m_mesh[mesh_id_t::cells_and_ghosts][level],
                                            m_mesh[mesh_id_t::cells][level-1])
                            .on(level);

                    std::array<double, 3> pred{{1./8, 0, -1./8}};
                    set([&](const auto& i, const auto& index)
                    {
                        auto j = index[0];
                        int sign_j = (j & 1)? -1: 1;

                        for(int ii=i.start; ii<i.end; ++ii)
                        {
                            auto i_cell = static_cast<PetscInt>(m_mesh.get_index(level, ii, j));
                            MatSetValue(A, i_cell, i_cell, 1, INSERT_VALUES);

                            int sign_i = (ii & 1)? -1: 1;

                            for(int is = -1; is<2; ++is)
                            {
                                // is = -1 --> parent_bottom --> -sign_j*pred[0]
                                // is =  0 --> parent        --> -sign_j*pred[1]
                                // is =  1 --> parent_top    --> -sign_j*pred[2]
                                auto i1 = static_cast<PetscInt>(m_mesh.get_index(level - 1, (ii>>1), (j>>1) + is));
                                MatSetValue(A, i_cell, i1, -sign_j*pred[static_cast<unsigned int>(is + 1)], INSERT_VALUES);

                                // is = -1 --> parent_left  --> -sign_i*pred[0]
                                // is =  0 --> parent       --> -sign_i*pred[1]
                                // is =  1 --> parent_right --> -sign_i*pred[2]
                                i1 = static_cast<PetscInt>(m_mesh.get_index(level - 1, (ii>>1) + is, (j>>1)));
                                MatSetValue(A, i_cell, i1, -sign_i*pred[static_cast<unsigned int>(is + 1)], INSERT_VALUES);
                            }

                            auto parent_bottom_left  = static_cast<PetscInt>(m_mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) - 1));
                            auto parent_bottom_right = static_cast<PetscInt>(m_mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) - 1));
                            auto parent_top_left     = static_cast<PetscInt>(m_mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) + 1));
                            auto parent_top_right    = static_cast<PetscInt>(m_mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) + 1));

                            MatSetValue(A, i_cell, parent_bottom_left , sign_i*sign_j*pred[0]*pred[0], INSERT_VALUES);
                            MatSetValue(A, i_cell, parent_bottom_right, sign_i*sign_j*pred[2]*pred[0], INSERT_VALUES);
                            MatSetValue(A, i_cell, parent_top_left    , sign_i*sign_j*pred[0]*pred[2], INSERT_VALUES);
                            MatSetValue(A, i_cell, parent_top_right   , sign_i*sign_j*pred[2]*pred[2], INSERT_VALUES);

                            auto i0 = static_cast<PetscInt>(m_mesh.get_index(level - 1, (ii>>1), (j>>1)));
                            MatSetValue(A, i_cell, i0, -1, INSERT_VALUES);
                        }
                    });
                }*/
            }

            void assemble_prediction_3D(Mat&)
            {
                for_each_prediction_ghost(m_mesh, [&](auto&)
                {
                    assert(false && "assemble_prediction_3D() not implemented.");
                });
            }


        public:
            template<class Func>
            static double L2Error(const Field& approximate, Func&& exact)
            {
                // In FV, we want only 1 quadrature point.
                // This is equivalent to 
                //       error += pow(exact(cell.center()) - approximate(cell.index), 2) * cell.length;
                GaussLegendre gl(0);

                double error_norm = 0;
                //double solution_norm = 0;
                for_each_cell(approximate.mesh(), [&](const auto& cell)
                {
                    error_norm += gl.quadrature<1>(cell, [&](const auto& point)
                    {
                        auto e = exact(point) - approximate[cell];
                        double norm_square;
                        if constexpr (Field::size == 1)
                        {
                            norm_square = e * e;
                        }
                        else
                        {
                            norm_square = xt::sum(e * e)();
                        }
                        return norm_square;
                    });

                    /*solution_norm += gl.quadrature<1>(cell, [&](const auto& point)
                    {
                        auto v = exact(point);
                        double v_square;
                        if constexpr (Field::size == 1)
                        {
                            v_square = v * v;
                        }
                        else
                        {
                            v_square = xt::sum(v * v)();
                        }
                        return v_square;
                    });*/
                });

                error_norm = sqrt(error_norm);
                //solution_norm = sqrt(solution_norm);
                //double relative_error = error_norm/solution_norm;
                //return relative_error;
                return error_norm;
            }
        };

    } // end namespace petsc
} // end namespace samurai