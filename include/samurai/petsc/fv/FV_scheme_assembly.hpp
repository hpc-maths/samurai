#pragma once
#include "../../boundary.hpp"
#include "../../schemes/fv/FV_scheme.hpp"
#include "../matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Finite Volume scheme.
         * This is the base class of CellBasedSchemeAssembly and FluxBasedSchemeAssembly.
         * It contains the management of
         *     - the boundary conditions
         *     - the projection/prediction ghosts
         *     - the unused ghosts
         */
        template <class Scheme>
        class FVSchemeAssembly : public MatrixAssembly
        {
          protected:

            using MatrixAssembly::m_col_shift;
            using MatrixAssembly::m_row_shift;

          public:

            using cfg_t                                            = typename Scheme::cfg_t;
            using bdry_cfg_t                                       = typename Scheme::bdry_cfg;
            using field_t                                          = typename Scheme::field_t;
            using mesh_t                                           = typename field_t::mesh_t;
            using mesh_id_t                                        = typename mesh_t::mesh_id_t;
            using interval_t                                       = typename mesh_t::interval_t;
            using field_value_type                                 = typename field_t::value_type; // double
            static constexpr std::size_t dim                       = field_t::dim;
            static constexpr std::size_t field_size                = field_t::size;
            static constexpr std::size_t output_field_size         = cfg_t::output_field_size;
            static constexpr std::size_t prediction_order          = mesh_t::config::prediction_order;
            static constexpr std::size_t bdry_neighbourhood_width  = bdry_cfg_t::neighbourhood_width;
            static constexpr std::size_t bdry_stencil_size         = bdry_cfg_t::stencil_size;
            static constexpr std::size_t nb_bdry_ghosts            = bdry_cfg_t::nb_ghosts;
            static constexpr DirichletEnforcement dirichlet_enfcmt = bdry_cfg_t::dirichlet_enfcmt;

            using dirichlet_t = Dirichlet<dim, interval_t, field_value_type, field_size>;
            using neumann_t   = Neumann<dim, interval_t, field_value_type, field_size>;

            using directional_bdry_config_t = DirectionalBoundaryConfig<field_t, output_field_size, bdry_stencil_size, nb_bdry_ghosts>;

          protected:

            const Scheme* m_scheme;
            field_t* m_unknown = nullptr;
            std::size_t m_n_cells;
            InsertMode m_current_insert_mode = INSERT_VALUES;
            std::vector<bool> m_is_row_empty;

          public:

            explicit FVSchemeAssembly(const Scheme& scheme)
                : m_scheme(&scheme)
                , m_unknown(&scheme.unknown())
            {
                this->set_name(scheme.name());
                reset();
            }

            auto& scheme() const
            {
                return *m_scheme;
            }

            void reset() override
            {
                m_n_cells = mesh().nb_cells();
                // std::cout << "reset " << this->name() << ", rows = " << matrix_rows() << std::endl;
                m_is_row_empty.resize(static_cast<std::size_t>(matrix_rows()));
                std::fill(m_is_row_empty.begin(), m_is_row_empty.end(), true);
            }

            void set_unknown(field_t& unknown)
            {
                m_unknown = &unknown;
            }

            auto& unknown() const
            {
                assert(m_unknown);
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
                else if constexpr (field_t::is_soa)
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
                else if constexpr (field_t::is_soa)
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
                else if constexpr (field_t::is_soa)
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
                else if constexpr (field_t::is_soa)
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
                                auto config = scheme().dirichlet_config(towards_out);
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
                                auto config = scheme().neumann_config(towards_out);
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

          public:

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
                                auto config = scheme().dirichlet_config(towards_out);
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
                                auto config = scheme().neumann_config(towards_out);
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
                                auto config = scheme().dirichlet_config(towards_out);
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
                                auto config = scheme().neumann_config(towards_out);
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

          public:

            void assemble_projection(Mat& A) override
            {
                static constexpr PetscInt number_of_children = (1 << dim);

                for_each_projection_ghost_and_children_cells<PetscInt>(
                    mesh(),
                    [&](auto level, PetscInt ghost, const std::array<PetscInt, number_of_children>& children)
                    {
                        double h       = cell_length(level);
                        double scaling = 1. / (h * h);
                        for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            PetscInt ghost_index = row_index(ghost, field_i);
                            MatSetValue(A, ghost_index, ghost_index, scaling, INSERT_VALUES);
                            for (unsigned int i = 0; i < number_of_children; ++i)
                            {
                                MatSetValue(A, ghost_index, col_index(children[i], field_i), -scaling / number_of_children, INSERT_VALUES);
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

          private:

            void assemble_prediction_1D(Mat& A)
            {
                using index_t = int;
                samurai::for_each_prediction_ghost(
                    mesh(),
                    [&](auto& ghost)
                    {
                        double h       = cell_length(ghost.level);
                        double scaling = 1. / (h * h);
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = this->row_index(ghost, field_i);
                            MatSetValue(A, ghost_index, ghost_index, scaling, INSERT_VALUES);

                            auto ii      = ghost.indices(0);
                            auto ig      = ii / 2;
                            double isign = (ii & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_order + 1>(isign);

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->mesh().get_index(ghost.level - 1, ig)), field_i);
                            MatSetValue(A, ghost_index, parent_index, -scaling, INSERT_VALUES);

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                if (ci != prediction_order)
                                {
                                    double value           = -interpx[ci];
                                    auto coarse_cell_index = this->col_index(
                                        static_cast<PetscInt>(
                                            this->mesh().get_index(ghost.level - 1, ig + static_cast<index_t>(ci - prediction_order))),
                                        field_i);
                                    MatSetValue(A, ghost_index, coarse_cell_index, scaling * value, INSERT_VALUES);
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
                        double h       = cell_length(ghost.level);
                        double scaling = 1. / (h * h);
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = this->row_index(ghost, field_i);
                            MatSetValue(A, ghost_index, ghost_index, scaling, INSERT_VALUES);

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
                            MatSetValue(A, ghost_index, parent_index, -scaling, INSERT_VALUES);

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
                                        MatSetValue(A, ghost_index, coarse_cell_index, scaling * value, INSERT_VALUES);
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
                        double h       = cell_length(ghost.level);
                        double scaling = 1. / (h * h);
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = this->row_index(ghost, field_i);
                            MatSetValue(A, ghost_index, ghost_index, scaling, INSERT_VALUES);

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
                            MatSetValue(A, ghost_index, parent_index, -scaling, INSERT_VALUES);

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
                                            MatSetValue(A, ghost_index, coarse_cell_index, scaling * value, INSERT_VALUES);
                                        }
                                    }
                                }
                            }
                            set_is_row_not_empty(ghost_index);
                        }
                    });
            }

          public:

            bool matrix_is_symmetric() const override
            {
                return scheme().matrix_is_symmetric(unknown());
            }

            bool matrix_is_spd() const override
            {
                return scheme().matrix_is_spd(unknown());
            }
        };

    } // end namespace petsc
} // end namespace samurai
