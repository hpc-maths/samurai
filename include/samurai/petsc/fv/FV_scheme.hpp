#pragma once
#include "../../boundary.hpp"
#include "../matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <PetscInt neighbourhood_width_, DirichletEnforcement dirichlet_enfcmt_ = Equation>
        struct BoundaryConfigFV
        {
            static constexpr PetscInt neighbourhood_width          = neighbourhood_width_;
            static constexpr PetscInt stencil_size                 = 1 + 2 * neighbourhood_width;
            static constexpr PetscInt nb_ghosts                    = neighbourhood_width;
            static constexpr DirichletEnforcement dirichlet_enfcmt = dirichlet_enfcmt_;
        };

        /**
         * Definition of one ghost equation to enforce the boundary condition.
         */
        template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size>
        struct BoundaryEquationCoeffs
        {
            static constexpr std::size_t field_size = Field::size;
            using field_value_type                  = typename Field::value_type; // double
            using coeffs_t                          = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;

            using stencil_coeffs_t = std::array<coeffs_t, bdry_stencil_size>;
            using rhs_coeffs_t     = coeffs_t;

            // Index of the ghost in the boundary stencil. The equation coefficients will be added on its row.
            std::size_t ghost_index;
            // Coefficients of the equation
            stencil_coeffs_t stencil_coeffs;
            // Coefficients of the right-hand side
            rhs_coeffs_t rhs_coeffs;
        };

        /**
         * Definition of one ghost equation to enforce the boundary condition.
         * It contains functions depending on h to get the equation coefficients.
         */
        template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size>
        struct BoundaryEquationConfig
        {
            using equation_coeffs_t         = BoundaryEquationCoeffs<Field, output_field_size, bdry_stencil_size>;
            using stencil_coeffs_t          = typename equation_coeffs_t::stencil_coeffs_t;
            using rhs_coeffs_t              = typename equation_coeffs_t::rhs_coeffs_t;
            using get_stencil_coeffs_func_t = std::function<stencil_coeffs_t(double)>;
            using get_rhs_coeffs_func_t     = std::function<rhs_coeffs_t(double)>;

            // Index of the ghost in the boundary stencil. The equation coefficients will be added on its row.
            std::size_t ghost_index;
            // Function to get the coefficients of the equation
            get_stencil_coeffs_func_t get_stencil_coeffs;
            // Function to get the coefficients of the right-hand side
            get_rhs_coeffs_func_t get_rhs_coeffs;
        };

        /**
         * For a boundary direction, defines one equation per boundary ghost.
         */
        template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size, std::size_t nb_bdry_ghosts>
        struct DirectionalBoundaryConfig
        {
            static constexpr std::size_t dim = Field::dim;
            using bdry_equation_config_t     = BoundaryEquationConfig<Field, output_field_size, bdry_stencil_size>;

            // Direction of the boundary and stencil for the computation of the boundary condition
            DirectionalStencil<bdry_stencil_size, dim> directional_stencil;
            // One equation per boundary ghost
            std::array<bdry_equation_config_t, nb_bdry_ghosts> equations;
        };

        /**
         * Finite Volume scheme.
         * This is the base class of CellBasedScheme and FluxBasedScheme.
         * It contains the management of
         *     - the boundary conditions
         *     - the projection/prediction ghosts
         *     - the unused ghosts
         */
        template <class Field, std::size_t output_field_size, class bdry_cfg>
        class FVScheme : public MatrixAssembly
        {
            template <class Scheme1, class Scheme2>
            friend class FluxBasedScheme_Sum_CellBasedScheme;

            template <class Scheme>
            friend class Scalar_x_FluxBasedScheme;

            template <std::size_t rows, std::size_t cols, class... Operators>
            friend class MonolithicBlockAssembly;

          protected:

            using MatrixAssembly::m_col_shift;
            using MatrixAssembly::m_row_shift;

          public:

            using Mesh                                             = typename Field::mesh_t;
            using mesh_id_t                                        = typename Mesh::mesh_id_t;
            using interval_t                                       = typename Mesh::interval_t;
            using field_value_type                                 = typename Field::value_type; // double
            static constexpr std::size_t dim                       = Field::dim;
            static constexpr std::size_t field_size                = Field::size;
            static constexpr std::size_t prediction_order          = Mesh::config::prediction_order;
            static constexpr std::size_t bdry_neighbourhood_width  = bdry_cfg::neighbourhood_width;
            static constexpr std::size_t bdry_stencil_size         = bdry_cfg::stencil_size;
            static constexpr std::size_t nb_bdry_ghosts            = bdry_cfg::nb_ghosts;
            static constexpr DirichletEnforcement dirichlet_enfcmt = bdry_cfg::dirichlet_enfcmt;

            using dirichlet_t = Dirichlet<dim, interval_t, field_value_type, field_size>;
            using neumann_t   = Neumann<dim, interval_t, field_value_type, field_size>;

            using directional_bdry_config_t = DirectionalBoundaryConfig<Field, output_field_size, bdry_stencil_size, nb_bdry_ghosts>;

          protected:

            Field* m_unknown;
            std::size_t m_n_cells;
            InsertMode m_current_insert_mode = INSERT_VALUES;
            std::vector<bool> m_is_row_empty;

          public:

            explicit FVScheme(Field& unknown)
                : m_unknown(&unknown)
            {
                reset();
            }

            void reset() override
            {
                m_n_cells = mesh().nb_cells();
                // std::cout << "reset " << this->name() << ", rows = " << matrix_rows() << std::endl;
                m_is_row_empty.resize(static_cast<std::size_t>(matrix_rows()));
                std::fill(m_is_row_empty.begin(), m_is_row_empty.end(), true);
            }

            auto& unknown() const
            {
                return *m_unknown;
            }

            auto& mesh() const
            {
                return unknown().mesh();
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

            InsertMode current_insert_mode() const
            {
                return m_current_insert_mode;
            }

            void set_current_insert_mode(InsertMode insert_mode)
            {
                m_current_insert_mode = insert_mode;
            }

            // Global data index
            inline PetscInt col_index(PetscInt cell_index, [[maybe_unused]] unsigned int field_j) const
            {
                if constexpr (field_size == 1)
                {
                    return m_col_shift + cell_index;
                }
                else if constexpr (Field::is_soa)
                {
                    return m_col_shift + static_cast<PetscInt>(field_j * m_n_cells) + cell_index;
                }
                else
                {
                    return m_col_shift + cell_index * static_cast<PetscInt>(field_size) + static_cast<PetscInt>(field_j);
                }
            }

            inline PetscInt row_index(PetscInt cell_index, [[maybe_unused]] unsigned int field_i) const
            {
                if constexpr (output_field_size == 1)
                {
                    return m_row_shift + cell_index;
                }
                else if constexpr (Field::is_soa)
                {
                    return m_row_shift + static_cast<PetscInt>(field_i * m_n_cells) + cell_index;
                }
                else
                {
                    return m_row_shift + cell_index * static_cast<PetscInt>(output_field_size) + static_cast<PetscInt>(field_i);
                }
            }

            template <class CellT>
            inline PetscInt col_index(const CellT& cell, [[maybe_unused]] unsigned int field_j) const
            {
                if constexpr (field_size == 1)
                {
                    return m_col_shift + static_cast<PetscInt>(cell.index);
                }
                else if constexpr (Field::is_soa)
                {
                    return m_col_shift + static_cast<PetscInt>(field_j * m_n_cells + cell.index);
                }
                else
                {
                    return m_col_shift + static_cast<PetscInt>(cell.index * field_size + field_j);
                }
            }

            template <class CellT>
            inline PetscInt row_index(const CellT& cell, [[maybe_unused]] unsigned int field_i) const
            {
                if constexpr (output_field_size == 1)
                {
                    return m_row_shift + static_cast<PetscInt>(cell.index);
                }
                else if constexpr (Field::is_soa)
                {
                    return m_row_shift + static_cast<PetscInt>(field_i * m_n_cells + cell.index);
                }
                else
                {
                    return m_row_shift + static_cast<PetscInt>(cell.index * output_field_size + field_i);
                }
            }

            template <class Coeffs>
            inline double cell_coeff(const Coeffs& coeffs,
                                     std::size_t cell_number_in_stencil,
                                     [[maybe_unused]] unsigned int field_i,
                                     [[maybe_unused]] unsigned int field_j) const
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

            template <class int_type>
            inline void set_is_row_not_empty(int_type row_number)
            {
                assert(row_number - m_row_shift >= 0);
                m_is_row_empty[static_cast<std::size_t>(row_number - m_row_shift)] = false;
            }

          protected:

            //-------------------------------------------------------------//
            //      Configuration of the BC stencils and equations         //
            //-------------------------------------------------------------//

            auto get_directional_stencil(const DirectionVector<dim>& direction) const
            {
                auto dir_stencils = directional_stencils<dim, bdry_neighbourhood_width>();
                for (std::size_t d = 0; d < 2 * dim; ++d)
                {
                    if (direction == dir_stencils[d].direction)
                    {
                        return dir_stencils[d];
                    }
                }
                assert(false);
                return dir_stencils[0];
            }

            virtual directional_bdry_config_t dirichlet_config(const DirectionVector<dim>& direction) const
            {
                using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;
                directional_bdry_config_t config;

                config.directional_stencil = get_directional_stencil(direction);

                if constexpr (bdry_neighbourhood_width == 1)
                {
                    static constexpr std::size_t cell          = 0;
                    static constexpr std::size_t interior_cell = 1;
                    static constexpr std::size_t ghost         = 2;

                    // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is
                    //                        [  1/2    1/2 ] = dirichlet_value
                    config.equations[0].ghost_index        = ghost;
                    config.equations[0].get_stencil_coeffs = [&](double)
                    {
                        std::array<coeffs_t, bdry_stencil_size> coeffs;
                        coeffs[cell]          = 0.5 * eye<coeffs_t>();
                        coeffs[ghost]         = 0.5 * eye<coeffs_t>();
                        coeffs[interior_cell] = zeros<coeffs_t>();
                        return coeffs;
                    };
                    config.equations[0].get_rhs_coeffs = [&](double)
                    {
                        coeffs_t coeffs = eye<coeffs_t>();
                        return coeffs;
                    };
                }

                return config;
            }

            virtual directional_bdry_config_t neumann_config(const DirectionVector<dim>& direction) const
            {
                using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;
                directional_bdry_config_t config;

                config.directional_stencil = get_directional_stencil(direction);

                if constexpr (bdry_neighbourhood_width == 1)
                {
                    static constexpr std::size_t cell          = 0;
                    static constexpr std::size_t interior_cell = 1;
                    static constexpr std::size_t ghost         = 2;

                    // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is
                    //                    [ 1/h  -1/h ] = neumann_value
                    config.equations[0].ghost_index        = ghost;
                    config.equations[0].get_stencil_coeffs = [&](double)
                    {
                        std::array<coeffs_t, bdry_stencil_size> coeffs;
                        coeffs[cell]          = -eye<coeffs_t>();
                        coeffs[ghost]         = eye<coeffs_t>();
                        coeffs[interior_cell] = zeros<coeffs_t>();
                        return coeffs;
                    };
                    config.equations[0].get_rhs_coeffs = [&](double h)
                    {
                        coeffs_t coeffs = h * eye<coeffs_t>();
                        return coeffs;
                    };
                }

                return config;
            }

          public:

            //-------------------------------------------------------------//
            //        Sparsity pattern of the boundary conditions          //
            //-------------------------------------------------------------//

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                if (unknown().get_bc().empty())
                {
                    std::cerr << "Failure to assemble to boundary conditions in the operator '" << this->name()
                              << "': no boundary condition attached to the field '" << unknown().name() << "'." << std::endl;
                    assert(false);
                    return;
                }
                // Iterate over the boundary conditions set by the user
                for (auto& bc : unknown().get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out = directions[d];

                        int number_of_one = xt::sum(xt::abs(towards_out))[0];
                        if (number_of_one == 1)
                        {
                            auto& boundary_cells   = boundary_cells_directions[d];
                            dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                            neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                            if (dirichlet)
                            {
                                auto config = dirichlet_config(towards_out);
                                for_each_stencil_on_boundary(mesh(),
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
                                for_each_stencil_on_boundary(mesh(),
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
            }

          protected:

            template <class CellList, class CoeffList>
            void
            sparsity_pattern_dirichlet_bc(std::vector<PetscInt>& nnz, CellList& cells, std::array<CoeffList, nb_bdry_ghosts>& equations) const
            {
                for (std::size_t e = 0; e < nb_bdry_ghosts; ++e)
                {
                    const auto& eq    = equations[e];
                    const auto& ghost = cells[eq.ghost_index];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        if constexpr (dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                            nnz[static_cast<std::size_t>(row_index(ghost, field_i))] = 1;
                        }
                        else
                        {
                            nnz[static_cast<std::size_t>(row_index(ghost, field_i))] = bdry_stencil_size;
                        }
                    }
                }
            }

            template <class CellList, class CoeffList>
            void
            sparsity_pattern_neumann_bc(std::vector<PetscInt>& nnz, CellList& cells, std::array<CoeffList, nb_bdry_ghosts>& equations) const
            {
                for (std::size_t e = 0; e < nb_bdry_ghosts; ++e)
                {
                    const auto& eq    = equations[e];
                    const auto& ghost = cells[eq.ghost_index];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        nnz[static_cast<std::size_t>(row_index(ghost, field_i))] = bdry_stencil_size;
                    }
                }
            }

            //-------------------------------------------------------------//
            //             Assemble the boundary conditions                //
            //-------------------------------------------------------------//

            void assemble_boundary_conditions(Mat& A) override
            {
                // std::cout << "assemble_boundary_conditions of " << this->name() << std::endl;
                if (current_insert_mode() == ADD_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(INSERT_VALUES);
                }

                // Iterate over the boundary conditions set by the user
                for (auto& bc : unknown().get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out = directions[d];

                        int number_of_one = xt::sum(xt::abs(towards_out))[0];
                        if (number_of_one == 1)
                        {
                            auto& boundary_cells   = boundary_cells_directions[d];
                            dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                            neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                            if (dirichlet)
                            {
                                auto config = dirichlet_config(towards_out);
                                for_each_stencil_on_boundary(mesh(),
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
                                for_each_stencil_on_boundary(mesh(),
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
            }

            template <class CellList, class CoeffList>
            void assemble_bc(Mat& A, CellList& cells, std::array<CoeffList, nb_bdry_ghosts>& equations)
            {
                for (std::size_t e = 0; e < nb_bdry_ghosts; ++e)
                {
                    auto eq                    = equations[e];
                    const auto& equation_ghost = cells[eq.ghost_index];
                    for (std::size_t c = 0; c < bdry_stencil_size; ++c)
                    {
                        for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            PetscInt equation_row = col_index(equation_ghost, field_i);
                            PetscInt col          = col_index(cells[c], field_i);

                            double coeff = cell_coeff(eq.stencil_coeffs, c, field_i, field_i);

                            if (coeff != 0)
                            {
                                if constexpr (dirichlet_enfcmt == DirichletEnforcement::Elimination)
                                {
                                }
                                else
                                {
                                    MatSetValue(A, equation_row, col, coeff, INSERT_VALUES);
                                }
                                set_is_row_not_empty(equation_row);
                            }
                        }
                    }
                }
            }

          public:

            //-------------------------------------------------------------//
            //   Enforce the boundary conditions on the right-hand side    //
            //-------------------------------------------------------------//

            virtual void enforce_bc(Vec& b) const
            {
                // std::cout << "enforce_bc of " << this->name() << std::endl;
                if (!this->is_block())
                {
                    PetscInt b_rows;
                    VecGetSize(b, &b_rows);
                    if (b_rows != this->matrix_cols())
                    {
                        std::cerr << "Operator '" << this->name() << "': the number of rows in vector (" << b_rows
                                  << ") does not equal the number of columns of the matrix (" << this->matrix_cols() << ")" << std::endl;
                        assert(false);
                        return;
                    }
                }

                // Iterate over the boundary conditions set by the user
                for (auto& bc : unknown().get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out = directions[d];

                        int number_of_one = xt::sum(xt::abs(towards_out))[0];
                        if (number_of_one == 1)
                        {
                            auto& boundary_cells   = boundary_cells_directions[d];
                            dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                            neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                            if (dirichlet)
                            {
                                auto config = dirichlet_config(towards_out);
                                for_each_stencil_on_boundary(mesh(),
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
                                for_each_stencil_on_boundary(mesh(),
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
            }

            template <class CellList, class CoeffList, class BoundaryCondition>
            void enforce_bc(Vec& b,
                            CellList& cells,
                            std::array<CoeffList, nb_bdry_ghosts>& equations,
                            const BoundaryCondition* bc,
                            const DirectionVector<dim>& towards_out) const
            {
                auto& cell          = cells[0];
                auto boundary_point = cell.face_center(towards_out);

                for (std::size_t e = 0; e < nb_bdry_ghosts; ++e)
                {
                    auto eq                    = equations[e];
                    const auto& equation_ghost = cells[eq.ghost_index];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt equation_row = row_index(equation_ghost, field_i);

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

                        if constexpr (dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                        }
                        else
                        {
                            VecSetValue(b, equation_row, coeff * bc_value, INSERT_VALUES);
                        }
                    }
                }
            }

          public:

            //-------------------------------------------------------------//
            //                      Useless ghosts                         //
            //-------------------------------------------------------------//

            void add_1_on_diag_for_useless_ghosts(Mat& A) override
            {
                if (current_insert_mode() == ADD_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(INSERT_VALUES);
                }
                // std::cout << "add_1_on_diag_for_useless_ghosts of " << this->name() << std::endl;
                for (std::size_t i = 0; i < m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        auto error = MatSetValue(A,
                                                 m_row_shift + static_cast<PetscInt>(i),
                                                 m_col_shift + static_cast<PetscInt>(i),
                                                 1,
                                                 INSERT_VALUES);
                        if (error)
                        {
                            assert(false);
                        }
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
                        VecSetValue(b, m_row_shift + static_cast<PetscInt>(i), 0, INSERT_VALUES);
                    }
                }
            }

            //-------------------------------------------------------------//
            //                  Projection / prediction                    //
            //-------------------------------------------------------------//

            void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const override
            {
                // ----  Projection stencil size
                // cell + 2^dim children --> 1+2=3 in 1D
                //                           1+4=5 in 2D
                static constexpr std::size_t proj_stencil_size = 1 + (1 << dim);

                for_each_projection_ghost(mesh(),
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                              {
                                                  nnz[static_cast<std::size_t>(row_index(ghost, field_i))] = proj_stencil_size;
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

                for_each_prediction_ghost(mesh(),
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                              {
                                                  nnz[static_cast<std::size_t>(row_index(ghost, field_i))] = pred_stencil_size;
                                              }
                                          });
            }

            virtual void enforce_projection_prediction(Vec& b) const
            {
                // Projection
                for_each_projection_ghost(mesh(),
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                                              {
                                                  VecSetValue(b, col_index(ghost, field_i), 0, INSERT_VALUES);
                                              }
                                          });

                // Prediction
                for_each_prediction_ghost(mesh(),
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                                              {
                                                  VecSetValue(b, col_index(ghost, field_i), 0, INSERT_VALUES);
                                              }
                                          });
            }

          private:

            void assemble_projection(Mat& A) override
            {
                static constexpr PetscInt number_of_children = (1 << dim);

                for_each_projection_ghost_and_children_cells<PetscInt>(
                    mesh(),
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
                            set_is_row_not_empty(ghost_index);
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
                    mesh(),
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = this->row_index(ghost, field_i);
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
                            set_is_row_not_empty(ghost_index);
                        }
                    });
            }

            void assemble_prediction_2D(Mat& A)
            {
                using index_t = int;
                samurai::for_each_prediction_ghost(
                    mesh(),
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = this->row_index(ghost, field_i);
                            MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                            auto ii      = ghost.indices(0);
                            auto ig      = ii / 2;
                            auto j       = ghost.indices(1);
                            auto jg      = j / 2;
                            double isign = (ii & 1) ? -1 : 1;
                            double jsign = (j & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_order + 1>(isign);
                            auto interpy = samurai::interp_coeffs<2 * prediction_order + 1>(jsign);

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->mesh().get_index(ghost.level - 1, ig, jg)),
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
                                            static_cast<PetscInt>(this->mesh().get_index(ghost.level - 1,
                                                                                         ig + static_cast<index_t>(ci - prediction_order),
                                                                                         jg + static_cast<index_t>(cj - prediction_order))),
                                            field_i);
                                        MatSetValue(A, ghost_index, coarse_cell_index, value, INSERT_VALUES);
                                    }
                                }
                            }
                            set_is_row_not_empty(ghost_index);
                        }
                    });
            }

            void assemble_prediction_3D(Mat& A)
            {
                using index_t = int;
                samurai::for_each_prediction_ghost(
                    mesh(),
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = this->row_index(ghost, field_i);
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

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->mesh().get_index(ghost.level - 1, ig, jg, kg)),
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
                                            auto coarse_cell_index = this->col_index(static_cast<PetscInt>(this->mesh().get_index(
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
                            set_is_row_not_empty(ghost_index);
                        }
                    });
            }
        };

    } // end namespace petsc
} // end namespace samurai
