#pragma once
#include "../boundary.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "matrix_assembly.hpp"

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
            static constexpr PetscInt output_field_size            = output_field_size_;
            static constexpr PetscInt neighbourhood_width          = neighbourhood_width_;
            static constexpr PetscInt scheme_stencil_size          = scheme_stencil_size_;
            static constexpr PetscInt center_index                 = center_index_;
            static constexpr PetscInt contiguous_indices_start     = contiguous_indices_start_;
            static constexpr PetscInt contiguous_indices_size      = contiguous_indices_size_;
            static constexpr DirichletEnforcement dirichlet_enfcmt = dirichlet_enfcmt_;
        };

        template <std::size_t dim, std::size_t output_field_size, std::size_t neighbourhood_width = 1, DirichletEnforcement dirichlet_enfcmt = Equation>
        using StarStencilFV = CellBasedAssemblyConfig<output_field_size,
                                                      neighbourhood_width,
                                                      // ----  Stencil size
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
                                                      1 + 2 * neighbourhood_width,
                                                      // ---- Method of Dirichlet condition enforcement
                                                      dirichlet_enfcmt>;

        template <std::size_t output_field_size, DirichletEnforcement dirichlet_enfcmt = Equation>
        using OneCellStencilFV = CellBasedAssemblyConfig<output_field_size,
                                                         // ----  Stencil size
                                                         // Only one cell:
                                                         1,
                                                         // ---- Index of the stencil center
                                                         // (as defined in center_only_stencil())
                                                         0,
                                                         // ---- Start index and size of contiguous cell indices
                                                         0,
                                                         0,
                                                         // ---- Method of Dirichlet condition enforcement
                                                         dirichlet_enfcmt>;

        template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size>
        struct BoundaryEquationCoeffs
        {
            static constexpr std::size_t field_size = Field::size;
            using field_value_type                  = typename Field::value_type; // double
            using local_matrix_t                    = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;

            using stencil_coeffs_t = std::array<local_matrix_t, bdry_stencil_size>;
            using rhs_coeffs_t     = local_matrix_t;

            std::size_t ghost_index;
            stencil_coeffs_t stencil_coeffs;
            rhs_coeffs_t rhs_coeffs;
        };

        template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size>
        struct BoundaryEquationConfig
        {
            /*static constexpr std::size_t field_size = Field::size;
            using field_value_type                  = typename Field::value_type; // double
            using local_matrix_t                    = typename detail::LocalMatrix<field_value_type,
                                                                output_field_size,
                                                                field_size>::Type; // 'double' if field_size = 1, 'xtensor' representing a
                                                                                                      // matrix otherwise*/
            using equation_coeffs_t         = BoundaryEquationCoeffs<Field, output_field_size, bdry_stencil_size>;
            using stencil_coeffs_t          = typename equation_coeffs_t::stencil_coeffs_t;
            using rhs_coeffs_t              = typename equation_coeffs_t::rhs_coeffs_t;
            using get_stencil_coeffs_func_t = std::function<stencil_coeffs_t(double)>;
            using get_rhs_coeffs_func_t     = std::function<rhs_coeffs_t(double)>;

            std::size_t ghost_index;
            get_stencil_coeffs_func_t get_stencil_coeffs;
            get_rhs_coeffs_func_t get_rhs_coeffs;
        };

        template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size, std::size_t nb_ghosts>
        struct DirectionalBoundaryConfig
        {
            static constexpr std::size_t dim = Field::dim;
            using bdry_equation_config_t     = BoundaryEquationConfig<Field, output_field_size, bdry_stencil_size>;

            DirectionalStencil<bdry_stencil_size, dim> directional_stencil;
            std::array<bdry_equation_config_t, nb_ghosts> equations;
        };

        template <class cfg, class Field>
        class CellBasedScheme : public MatrixAssembly
        {
            template <class Scheme1, class Scheme2>
            friend class FluxBasedScheme_Sum_CellBasedScheme;

          public:

            using cfg_t   = cfg;
            using field_t = Field;

            using Mesh                                       = typename Field::mesh_t;
            using mesh_id_t                                  = typename Mesh::mesh_id_t;
            using interval_t                                 = typename Mesh::interval_t;
            using field_value_type                           = typename Field::value_type; // double
            static constexpr std::size_t field_size          = Field::size;
            static constexpr std::size_t output_field_size   = cfg::output_field_size;
            static constexpr std::size_t neighbourhood_width = cfg::neighbourhood_width;
            using local_matrix_t                             = typename detail::LocalMatrix<field_value_type,
                                                                output_field_size,
                                                                field_size>::Type; // 'double' if field_size = 1, 'xtensor' representing a
                                                                                                               // matrix otherwise
            static constexpr std::size_t dim               = Mesh::dim;
            static constexpr std::size_t prediction_order  = Mesh::config::prediction_order;
            static constexpr std::size_t bdry_stencil_size = 1 + 2 * neighbourhood_width;

            using stencil_t         = Stencil<cfg::scheme_stencil_size, dim>;
            using get_coeffs_func_t = std::function<std::array<local_matrix_t, cfg::scheme_stencil_size>(double)>;

            using dirichlet_t = Dirichlet<dim, interval_t, field_value_type, field_size>;
            using neumann_t   = Neumann<dim, interval_t, field_value_type, field_size>;

            using directional_bdry_config_t = DirectionalBoundaryConfig<Field, output_field_size, bdry_stencil_size, neighbourhood_width>;

            using MatrixAssembly::assemble_matrix;

          protected:

            Field& m_unknown;
            Mesh& m_mesh;
            std::size_t m_n_cells;
            stencil_t m_stencil;
            get_coeffs_func_t m_get_coefficients;
            std::vector<bool> m_is_row_empty;

          public:

            CellBasedScheme(Field& unknown, stencil_t s, get_coeffs_func_t get_coeffs)
                : m_unknown(unknown)
                , m_mesh(unknown.mesh())
                , m_stencil(s)
                , m_get_coefficients(get_coeffs)
            {
                reset();
            }

            void reset()
            {
                m_n_cells      = m_mesh.nb_cells();
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

            PetscInt matrix_rows() const override
            {
                return static_cast<PetscInt>(m_n_cells * output_field_size);
            }

            PetscInt matrix_cols() const override
            {
                return static_cast<PetscInt>(m_n_cells * field_size);
            }

          protected:

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

            template <class Coeffs>
            inline double
            cell_coeff(const Coeffs& coeffs, std::size_t cell_number_in_stencil, unsigned int field_i, unsigned int field_j) const
            {
                if constexpr (field_size == 1 && output_field_size == 1)
                {
                    return coeffs[cell_number_in_stencil];
                }
                else
                {
                    return coeffs[cell_number_in_stencil](field_i, field_j);
                }
            }

            template <class Coeffs>
            inline double rhs_coeff(const Coeffs& coeffs, unsigned int field_i, unsigned int field_j) const
            {
                if constexpr (field_size == 1 && output_field_size == 1)
                {
                    return coeffs;
                }
                else
                {
                    return coeffs(field_i, field_j);
                }
            }

          public:

            //-------------------------------------------------------------//
            //                     Sparsity pattern                        //
            //-------------------------------------------------------------//

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
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
                                      nnz[row_index(cell, field_i)] = scheme_nnz_i;
                                  });
                }
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                // Iterate over the boundary conditions set by the user
                for (auto& bc : m_unknown.get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out    = directions[d];
                        auto& boundary_cells = boundary_cells_directions[d];

                        dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                        neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                        if (dirichlet)
                        {
                            auto config = dirichlet_config(towards_out);
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         config.directional_stencil.stencil,
                                                         config.equations,
                                                         [&](auto& cells, auto& equations)
                                                         {
                                                             sparsity_pattern_dirichlet_bc(nnz, cells, equations);
                                                         });
                        }
                        else if (neumann)
                        {
                            auto config = neumann_config(towards_out);
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         config.directional_stencil.stencil,
                                                         config.equations,
                                                         [&](auto& cells, auto& equations)
                                                         {
                                                             sparsity_pattern_neumann_bc(nnz, cells, equations);
                                                         });
                        }
                        else
                        {
                            std::cerr << "Unknown boundary condition type" << std::endl;
                        }
                    }
                }
            }

          protected:

            template <class CellList, class CoeffList>
            void sparsity_pattern_dirichlet_bc(std::vector<PetscInt>& nnz,
                                               CellList& cells,
                                               std::array<CoeffList, neighbourhood_width>& equations) const
            {
                for (std::size_t e = 0; e < neighbourhood_width; ++e)
                {
                    const auto& eq    = equations[e];
                    const auto& ghost = cells[eq.ghost_index];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                            nnz[row_index(ghost, field_i)] = 1;
                        }
                        else
                        {
                            nnz[row_index(ghost, field_i)] = bdry_stencil_size;
                        }
                    }
                }
            }

            template <class CellList, class CoeffList>
            void
            sparsity_pattern_neumann_bc(std::vector<PetscInt>& nnz, CellList& cells, std::array<CoeffList, neighbourhood_width>& equations) const
            {
                for (std::size_t e = 0; e < neighbourhood_width; ++e)
                {
                    const auto& eq    = equations[e];
                    const auto& ghost = cells[eq.ghost_index];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        nnz[row_index(ghost, field_i)] = bdry_stencil_size;
                    }
                }
            }

          public:

            void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const override
            {
                // ----  Projection stencil size
                // cell + 2^dim children --> 1+2=3 in 1D
                //                           1+4=5 in 2D
                static constexpr std::size_t proj_stencil_size = 1 + (1 << dim);

                for_each_projection_ghost(m_mesh,
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                              {
                                                  nnz[row_index(ghost, field_i)] = proj_stencil_size;
                                              }
                                          });
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& nnz) const override
            {
                // ----  Prediction stencil size
                // Order 1: cell + hypercube of 3 coarser cells --> 1 + 3= 4 in 1D
                //                                                  1 + 9=10 in 2D
                // Order 2: cell + hypercube of 5 coarser cells --> 1 + 5= 6 in 1D
                //                                                  1 +25=21 in 2D
                static constexpr std::size_t pred_stencil_size = 1 + ce_pow(2 * prediction_order + 1, dim);

                for_each_prediction_ghost(m_mesh,
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                              {
                                                  nnz[row_index(ghost, field_i)] = pred_stencil_size;
                                              }
                                          });
            }

          protected:

            //-------------------------------------------------------------//
            //             Assemble scheme in the interior                 //
            //-------------------------------------------------------------//

            void assemble_scheme(Mat& A) override
            {
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

            //-------------------------------------------------------------//
            //             Assemble the boundary conditions                //
            //-------------------------------------------------------------//

            void assemble_boundary_conditions(Mat& A) override
            {
                if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                {
                    // Must flush to use ADD_VALUES instead of INSERT_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                }

                // Iterate over the boundary conditions set by the user
                for (auto& bc : m_unknown.get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out    = directions[d];
                        auto& boundary_cells = boundary_cells_directions[d];

                        dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                        neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                        if (dirichlet)
                        {
                            auto config = dirichlet_config(towards_out);
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         config.directional_stencil.stencil,
                                                         config.equations,
                                                         [&](auto& cells, auto& equations)
                                                         {
                                                             assemble_bc(A, cells, equations);
                                                         });
                        }
                        else if (neumann)
                        {
                            auto config = neumann_config(towards_out);
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         config.directional_stencil.stencil,
                                                         config.equations,
                                                         [&](auto& cells, auto& equations)
                                                         {
                                                             assemble_bc(A, cells, equations);
                                                         });
                        }
                        else
                        {
                            std::cerr << "Unknown boundary condition type" << std::endl;
                        }
                    }
                }
            }

            virtual directional_bdry_config_t dirichlet_config(const DirectionVector<dim>& direction) const
            {
                using local_matrix_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::local_matrix_t;
                directional_bdry_config_t config;

                auto dir_stencils = directional_stencils<dim, neighbourhood_width>();
                bool found        = false;
                for (std::size_t d = 0; d < 2 * dim; ++d)
                {
                    if (direction == dir_stencils[d].direction)
                    {
                        found                      = true;
                        config.directional_stencil = dir_stencils[d];
                        break;
                    }
                }
                assert(found);

                if constexpr (neighbourhood_width == 1)
                {
                    static constexpr std::size_t cell          = 0;
                    static constexpr std::size_t interior_cell = 1;
                    static constexpr std::size_t ghost         = 2;

                    // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is
                    //                        [  1/2    1/2 ] = dirichlet_value
                    // which is equivalent to
                    //                        [  1/h2   1/h2] = 2 * 1/h2 * dirichlet_value
                    config.equations[0].ghost_index        = ghost;
                    config.equations[0].get_stencil_coeffs = [&](double h)
                    {
                        std::array<local_matrix_t, bdry_stencil_size> coeffs;
                        auto Identity         = eye<local_matrix_t>();
                        coeffs[cell]          = 1 / (h * h) * Identity;
                        coeffs[ghost]         = 1 / (h * h) * Identity;
                        coeffs[interior_cell] = zeros<local_matrix_t>();
                        return coeffs;
                    };
                    config.equations[0].get_rhs_coeffs = [&](double h)
                    {
                        local_matrix_t coeffs;
                        auto Identity = eye<local_matrix_t>();
                        coeffs        = 2 / (h * h) * Identity;
                        return coeffs;
                    };
                }
                if constexpr (neighbourhood_width == 2)
                {
                }

                return config;
            }

            virtual directional_bdry_config_t neumann_config(const DirectionVector<dim>& direction) const
            {
                using local_matrix_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::local_matrix_t;
                directional_bdry_config_t config;

                auto dir_stencils = directional_stencils<dim, neighbourhood_width>();
                bool found        = false;
                for (std::size_t d = 0; d < 2 * dim; ++d)
                {
                    if (direction == dir_stencils[d].direction)
                    {
                        found                      = true;
                        config.directional_stencil = dir_stencils[d];
                        break;
                    }
                }
                assert(found);

                if constexpr (neighbourhood_width == 1)
                {
                    static constexpr std::size_t cell          = 0;
                    static constexpr std::size_t interior_cell = 1;
                    static constexpr std::size_t ghost         = 2;

                    // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is
                    //                    [ 1/h  -1/h ] = neumann_value
                    // However, to have symmetry, we want to have 1/h2 as the off-diagonal coefficient, so
                    //                    [1/h2  -1/h2] = (1/h) * neumann_value
                    config.equations[0].ghost_index        = ghost;
                    config.equations[0].get_stencil_coeffs = [&](double h)
                    {
                        std::array<local_matrix_t, bdry_stencil_size> coeffs;
                        double one_over_h2    = 1 / (h * h);
                        auto Identity         = eye<local_matrix_t>();
                        coeffs[cell]          = -one_over_h2 * Identity;
                        coeffs[ghost]         = one_over_h2 * Identity;
                        coeffs[interior_cell] = zeros<local_matrix_t>();
                        return coeffs;
                    };
                    config.equations[0].get_rhs_coeffs = [&](double h)
                    {
                        auto Identity         = eye<local_matrix_t>();
                        local_matrix_t coeffs = (1 / h) * Identity;
                        return coeffs;
                    };
                }
                if constexpr (neighbourhood_width == 2)
                {
                }

                return config;
            }

            /*template <class CellList, class CoeffList>
            virtual void assemble_dirichlet_bc(Mat& A, CellList& cells, CoeffList& coeffs)
            {
                const auto& cell = cells[0];
                for (std::size_t g = 1; g < 2; ++g)
                {
                    const auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                        double coeff = cell_coeff(coeffs, g, field_i, field_i);

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

                            // the coeff of the ghost is removed from the stencil (we want 0 so we substract the coeff we set before)
                            MatSetValue(A, cell_index, ghost_index, -coeff, ADD_VALUES);
                            // the coeff is substracted from the center of the stencil
                            MatSetValue(A, cell_index, cell_index, -coeff, ADD_VALUES);
                            // 1 is added to the diagonal of the ghost
                            MatSetValue(A, ghost_index, ghost_index, 1, ADD_VALUES);
                        }
                        else
                        {
                            coeff = coeff == 0 ? 1 : coeff;
                            // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is
                            //                        [  1/2    1/2 ] = dirichlet_value
                            // which is equivalent to
                            //                        [-coeff -coeff] = -2 * coeff * dirichlet_value
                            MatSetValue(A, ghost_index, ghost_index, -coeff, INSERT_VALUES);
                            MatSetValue(A, ghost_index, cell_index, -coeff, INSERT_VALUES);
                        }
                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                }
            }*/
            template <class CellList, class CoeffList>
            void assemble_bc(Mat& A, CellList& cells, std::array<CoeffList, neighbourhood_width>& equations)
            {
                for (std::size_t e = 0; e < neighbourhood_width; ++e)
                {
                    auto eq                    = equations[e];
                    const auto& equation_ghost = cells[eq.ghost_index];
                    for (std::size_t c = 0; c < bdry_stencil_size; ++c)
                    {
                        for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            PetscInt equation_row = static_cast<PetscInt>(col_index(equation_ghost, field_i));
                            PetscInt col          = static_cast<PetscInt>(col_index(cells[c], field_i));

                            double coeff = cell_coeff(eq.stencil_coeffs, c, field_i, field_i);

                            if (coeff != 0)
                            {
                                if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                                {
                                }
                                else
                                {
                                    MatSetValue(A, equation_row, col, coeff, INSERT_VALUES);
                                }
                                m_is_row_empty[static_cast<std::size_t>(equation_row)] = false;
                            }
                        }
                    }
                }
            }

            /*template <class CellList, class CoeffList>
            virtual void assemble_neumann_bc(Mat& A, CellList& cells, CoeffList& coeffs)
            {
                const auto& cell = cells[0];
                for (std::size_t g = 1; g < 2; ++g)
                {
                    const auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                        double coeff = cell_coeff(coeffs, g, field_i, field_i);
                        coeff        = coeff == 0 ? 1 : coeff;
                        // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is
                        //                    [ 1/h  -1/h ] = neumann_value
                        // However, to have symmetry, we want to have coeff as the off-diagonal coefficient, so
                        //                   [-coeff coeff] = -coeff * h * neumann_value
                        if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                            MatSetValue(A, ghost_index, ghost_index, -coeff, ADD_VALUES);
                            MatSetValue(A, ghost_index, cell_index, coeff, ADD_VALUES);
                        }
                        else
                        {
                            MatSetValue(A, ghost_index, ghost_index, -coeff, INSERT_VALUES);
                            MatSetValue(A, ghost_index, cell_index, coeff, INSERT_VALUES);
                        }

                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                }
            }*/

          public:

            //-------------------------------------------------------------//
            //   Enforce the boundary conditions on the right-hand side    //
            //-------------------------------------------------------------//

            virtual void enforce_bc(Vec& b) const
            {
                // Iterate over the boundary conditions set by the user
                for (auto& bc : m_unknown.get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out    = directions[d];
                        auto& boundary_cells = boundary_cells_directions[d];

                        // Stencil<2, dim> stencil = in_out_stencil<dim>(towards_out);

                        dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                        neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                        if (dirichlet)
                        {
                            auto config = dirichlet_config(towards_out);
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         config.directional_stencil.stencil,
                                                         config.equations,
                                                         [&](auto& cells, auto& equations)
                                                         {
                                                             enforce_bc(b, cells, equations, dirichlet, towards_out);
                                                         });
                        }
                        else if (neumann)
                        {
                            auto config = neumann_config(towards_out);
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         config.directional_stencil.stencil,
                                                         config.equations,
                                                         [&](auto& cells, auto& equations)
                                                         {
                                                             enforce_bc(b, cells, equations, neumann, towards_out);
                                                         });
                        }
                        else
                        {
                            std::cerr << "Unknown boundary condition type" << std::endl;
                        }
                    }
                }
            }

            /*template <class CellList, class CoeffList>
            void enforce_dirichlet_bc(Vec& b,
                                      CellList& cells,
                                      CoeffList& coeffs,
                                      const dirichlet_t& dirichlet,
                                      const DirectionVector<dim>& towards_out) const
            {
                auto& cell          = cells[0];
                auto boundary_point = cell.face_center(towards_out);

                for (std::size_t g = 1; g < 2; ++g)
                {
                    const auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                        double coeff = cell_coeff(coeffs, g, field_i, field_i);

                        double dirichlet_value;
                        if constexpr (field_size == 1)
                        {
                            dirichlet_value = dirichlet.value(boundary_point);
                        }
                        else
                        {
                            dirichlet_value = dirichlet.value(boundary_point)(field_i); // TODO: call get_value() only once instead of once
                                                                                        // per field_i
                        }

                        if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                            VecSetValue(b, cell_index, -2 * coeff * dirichlet_value, ADD_VALUES);
                        }
                        else
                        {
                            coeff = coeff == 0 ? 1 : coeff;
                            VecSetValue(b, ghost_index, -2 * coeff * dirichlet_value, INSERT_VALUES);
                        }
                    }
                }
            }

            template <class CellList, class CoeffList>
            void
            enforce_neumann_bc(Vec& b, CellList& cells, CoeffList& coeffs, const neumann_t& neumann, const DirectionVector<dim>&
            towards_out) const
            {
                auto& cell          = cells[0];
                auto boundary_point = cell.face_center(towards_out);

                for (std::size_t g = 1; g < 2; ++g)
                {
                    const auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        // PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                        double coeff = cell_coeff(coeffs, g, field_i, field_i);

                        coeff   = coeff == 0 ? 1 : coeff;
                        auto& h = cell.length;
                        double neumann_value;
                        if constexpr (field_size == 1)
                        {
                            neumann_value = neumann.value(boundary_point);
                        }
                        else
                        {
                            neumann_value = neumann.value(boundary_point)(field_i); // TODO: call get_value() only once instead of once per
                                                                                    // field_i
                        }
                        VecSetValue(b, ghost_index, -coeff * h * neumann_value, INSERT_VALUES);
                    }
                }
            }*/

            template <class CellList, class CoeffList, class BoundaryCondition>
            void enforce_bc(Vec& b,
                            CellList& cells,
                            std::array<CoeffList, neighbourhood_width>& equations,
                            const BoundaryCondition* bc,
                            const DirectionVector<dim>& towards_out) const
            {
                auto& cell          = cells[0];
                auto boundary_point = cell.face_center(towards_out);

                for (std::size_t e = 0; e < neighbourhood_width; ++e)
                {
                    auto eq                    = equations[e];
                    const auto& equation_ghost = cells[eq.ghost_index];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt equation_row = static_cast<PetscInt>(col_index(equation_ghost, field_i));

                        double coeff = rhs_coeff(eq.rhs_coeffs, field_i, field_i);
                        assert(coeff != 0);

                        double bc_value;
                        if constexpr (field_size == 1)
                        {
                            bc_value = bc->value(boundary_point);
                        }
                        else
                        {
                            bc_value = bc->value(boundary_point)(field_i); // TODO: call get_value() only once instead of
                                                                           // once per field_i
                        }

                        if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                        }
                        else
                        {
                            VecSetValue(b, equation_row, coeff * bc_value, INSERT_VALUES);
                        }
                    }
                }
            }

            //-------------------------------------------------------------//
            //                      Useless ghosts                         //
            //-------------------------------------------------------------//

            void add_1_on_diag_for_useless_ghosts(Mat& A) override
            {
                for (std::size_t i = 0; i < m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        MatSetValue(A, static_cast<PetscInt>(i), static_cast<PetscInt>(i), 1, INSERT_VALUES);
                        // m_is_row_empty[i] = false;
                    }
                }
            }

            void add_0_for_useless_ghosts(Vec& b) const
            {
                for (std::size_t i = 0; i < m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        VecSetValue(b, static_cast<PetscInt>(i), 0, INSERT_VALUES);
                    }
                }
            }

            //-------------------------------------------------------------//
            //                  Projection / prediction                    //
            //-------------------------------------------------------------//

            virtual void enforce_projection_prediction(Vec& b) const
            {
                // Projection
                for_each_projection_ghost(m_mesh,
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                                              {
                                                  VecSetValue(b, static_cast<PetscInt>(col_index(ghost, field_i)), 0, INSERT_VALUES);
                                              }
                                          });

                // Prediction
                for_each_prediction_ghost(m_mesh,
                                          [&](auto& ghost)
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

                for_each_projection_ghost_and_children_cells<PetscInt>(
                    m_mesh,
                    [&](PetscInt ghost, const std::array<PetscInt, number_of_children>& children)
                    {
                        for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            PetscInt ghost_index = row_index(ghost, field_i);
                            MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);
                            for (unsigned int i = 0; i < number_of_children; ++i)
                            {
                                MatSetValue(A, ghost_index, col_index(children[i], field_i), -1. / number_of_children, INSERT_VALUES);
                            }
                            m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                        }
                    });
            }

            void assemble_prediction(Mat& A) override
            {
                static_assert(dim >= 1 && dim <= 3,
                              "assemble_prediction() is not implemented for "
                              "this dimension.");
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
                using index_t = int;
                samurai::for_each_prediction_ghost(
                    m_mesh,
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = static_cast<PetscInt>(this->row_index(ghost, field_i));
                            MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                            auto ii      = ghost.indices(0);
                            auto ig      = ii / 2;
                            double isign = (ii & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_order + 1>(isign);

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1, ig)), field_i);
                            MatSetValue(A, ghost_index, parent_index, -1, INSERT_VALUES);

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                if (ci != prediction_order)
                                {
                                    double value           = -interpx[ci];
                                    auto coarse_cell_index = this->col_index(
                                        static_cast<PetscInt>(
                                            this->m_mesh.get_index(ghost.level - 1, ig + static_cast<index_t>(ci - prediction_order))),
                                        field_i);
                                    MatSetValue(A, ghost_index, coarse_cell_index, value, INSERT_VALUES);
                                }
                            }
                            this->m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                        }
                    });
            }

            void assemble_prediction_2D(Mat& A)
            {
                using index_t = int;
                samurai::for_each_prediction_ghost(
                    m_mesh,
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = static_cast<PetscInt>(this->row_index(ghost, field_i));
                            MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                            auto ii      = ghost.indices(0);
                            auto ig      = ii / 2;
                            auto j       = ghost.indices(1);
                            auto jg      = j / 2;
                            double isign = (ii & 1) ? -1 : 1;
                            double jsign = (j & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_order + 1>(isign);
                            auto interpy = samurai::interp_coeffs<2 * prediction_order + 1>(jsign);

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1, ig, jg)),
                                                                field_i);
                            MatSetValue(A, ghost_index, parent_index, -1, INSERT_VALUES);

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                                {
                                    if (ci != prediction_order || cj != prediction_order)
                                    {
                                        double value           = -interpx[ci] * interpy[cj];
                                        auto coarse_cell_index = this->col_index(
                                            static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1,
                                                                                         ig + static_cast<index_t>(ci - prediction_order),
                                                                                         jg + static_cast<index_t>(cj - prediction_order))),
                                            field_i);
                                        MatSetValue(A, ghost_index, coarse_cell_index, value, INSERT_VALUES);
                                    }
                                }
                            }
                            this->m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                        }
                    });
            }

            void assemble_prediction_3D(Mat& A)
            {
                using index_t = int;
                samurai::for_each_prediction_ghost(
                    m_mesh,
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = static_cast<PetscInt>(this->row_index(ghost, field_i));
                            MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                            auto ii      = ghost.indices(0);
                            auto ig      = ii / 2;
                            auto j       = ghost.indices(1);
                            auto jg      = j / 2;
                            auto k       = ghost.indices(2);
                            auto kg      = k / 2;
                            double isign = (ii & 1) ? -1 : 1;
                            double jsign = (j & 1) ? -1 : 1;
                            double ksign = (k & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_order + 1>(isign);
                            auto interpy = samurai::interp_coeffs<2 * prediction_order + 1>(jsign);
                            auto interpz = samurai::interp_coeffs<2 * prediction_order + 1>(ksign);

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1, ig, jg, kg)),
                                                                field_i);
                            MatSetValue(A, ghost_index, parent_index, -1, INSERT_VALUES);

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                                {
                                    for (std::size_t ck = 0; ck < interpz.size(); ++ck)
                                    {
                                        if (ci != prediction_order || cj != prediction_order || ck != prediction_order)
                                        {
                                            double value           = -interpx[ci] * interpy[cj] * interpz[ck];
                                            auto coarse_cell_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(
                                                                                         ghost.level - 1,
                                                                                         ig + static_cast<index_t>(ci - prediction_order),
                                                                                         jg + static_cast<index_t>(cj - prediction_order),
                                                                                         kg + static_cast<index_t>(ck - prediction_order))),
                                                                                     field_i);
                                            MatSetValue(A, ghost_index, coarse_cell_index, value, INSERT_VALUES);
                                        }
                                    }
                                }
                            }
                            this->m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
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
