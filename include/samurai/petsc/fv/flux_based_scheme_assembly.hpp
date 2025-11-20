#pragma once
#include "../../interface.hpp"
#include "../../schemes/fv/flux_based/flux_based_scheme.hpp"
#include "FV_scheme_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Assembly for LINEAR schemes
         */
        template <class Scheme>
        class Assembly<Scheme, std::enable_if_t<is_FluxBasedScheme_v<Scheme>>> : public FVSchemeAssembly<Scheme>
        {
          protected:

            using base_class = FVSchemeAssembly<Scheme>;
            using base_class::dim;
            using base_class::input_n_comp;
            using base_class::is_locally_owned;
            using base_class::output_n_comp;
            using base_class::set_is_row_not_empty;

          public:

            using base_class::ghost_elimination_enabled;
            using base_class::global_col_index;
            using base_class::global_row_index;
            using base_class::local_col_index;
            using base_class::local_row_index;
            using base_class::mesh;
            using base_class::scheme;
            using base_class::set_current_insert_mode;
            using base_class::unknown;

            using scheme_t                            = Scheme;
            using cfg_t                               = typename Scheme::cfg_t;
            using bdry_cfg_t                          = typename Scheme::bdry_cfg;
            using field_t                             = typename Scheme::field_t;
            static constexpr std::size_t stencil_size = cfg_t::stencil_size;

          private:

            bool m_include_boundary_fluxes = true;

          public:

            explicit Assembly(const Scheme& s)
                : base_class(s)
            {
                set_current_insert_mode(ADD_VALUES);
                m_include_boundary_fluxes = s.include_boundary_fluxes();
            }

            void include_boundary_fluxes(bool include)
            {
                m_include_boundary_fluxes = include;
            }

            //-------------------------------------------------------------//
            //                     Sparsity pattern                        //
            //-------------------------------------------------------------//

            void sparsity_pattern_scheme(std::vector<PetscInt>& d_nnz, [[maybe_unused]] std::vector<PetscInt>& o_nnz) const override
            {
                // std::cout << "[" << mpi::communicator().rank() << "] sparsity_pattern_scheme() of interior interfaces" << std::endl;
                auto& flux_def = scheme().flux_definition();
                for (std::size_t d = 0; d < dim; ++d)
                {
                    for_each_interior_interface<Run::Sequential, Get::Cells, /* include_periodic = */ false>(
                        mesh(),
                        flux_def[d].direction,
                        flux_def[d].stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
#if SAMURAI_WITH_MPI
                            bool cell_0_locally_owned = is_locally_owned(interface_cells[0]);
                            bool cell_1_locally_owned = is_locally_owned(interface_cells[1]);
#endif
                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
                                std::size_t row_cell_0 = static_cast<std::size_t>(local_row_index(interface_cells[0], field_i));
                                std::size_t row_cell_1 = static_cast<std::size_t>(local_row_index(interface_cells[1], field_i));

#ifdef SAMURAI_WITH_MPI
                                for (std::size_t c = 0; c < stencil_size; ++c)
                                {
                                    if (is_locally_owned(comput_cells[c]))
                                    {
                                        if (cell_0_locally_owned)
                                        {
                                            d_nnz.at(row_cell_0) += input_n_comp;
                                        }
                                        if (cell_1_locally_owned)
                                        {
                                            d_nnz.at(row_cell_1) += input_n_comp;
                                        }
                                    }
                                    else
                                    {
                                        if (cell_0_locally_owned)
                                        {
                                            o_nnz.at(row_cell_0) += input_n_comp;
                                        }
                                        if (cell_1_locally_owned)
                                        {
                                            o_nnz.at(row_cell_1) += input_n_comp;
                                        }
                                    }
                                }
#else
                                if constexpr (ghost_elimination_enabled)
                                {
                                    for (std::size_t c = 0; c < stencil_size; ++c)
                                    {
                                        auto it_ghost = this->m_ghost_recursion.find(comput_cells[c].index);
                                        if (it_ghost == this->m_ghost_recursion.end())
                                        {
                                            d_nnz[row_cell_0] += static_cast<PetscInt>(input_n_comp);
                                            d_nnz[row_cell_1] += static_cast<PetscInt>(input_n_comp);
                                        }
                                        else
                                        {
                                            auto& linear_comb = it_ghost->second;
                                            d_nnz[row_cell_0] += static_cast<PetscInt>(linear_comb.size() * input_n_comp);
                                            d_nnz[row_cell_1] += static_cast<PetscInt>(linear_comb.size() * input_n_comp);
                                        }
                                    }
                                }
                                else
                                {
                                    d_nnz[row_cell_0] += static_cast<PetscInt>(stencil_size * input_n_comp);
                                    d_nnz[row_cell_1] += static_cast<PetscInt>(stencil_size * input_n_comp);
                                }
#endif
                            }
                        });

                    if (m_include_boundary_fluxes)
                    {
                        // std::cout << "[" << mpi::communicator().rank() << "] sparsity_pattern_scheme() of boundary interfaces" <<
                        // std::endl;
                        for_each_boundary_interface__both_directions(
                            mesh(),
                            flux_def[d].direction,
                            flux_def[d].stencil,
                            [&](auto& cell, [[maybe_unused]] auto& comput_cells)
                            {
                                if (!is_locally_owned(cell))
                                {
                                    return;
                                }
                                for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                                {
                                    std::size_t row_cell = static_cast<std::size_t>(local_row_index(cell, field_i));
#ifdef SAMURAI_WITH_MPI
                                    for (std::size_t c = 0; c < stencil_size; ++c)
                                    {
                                        if (is_locally_owned(comput_cells[c]))
                                        {
                                            d_nnz.at(row_cell) += input_n_comp;
                                        }
                                        else
                                        {
                                            o_nnz.at(row_cell) += input_n_comp;
                                        }
                                    }
#else
                                    d_nnz[row_cell] += static_cast<PetscInt>(stencil_size * input_n_comp);
#endif
                                }
                            });
                    }
                }
            }

            //-------------------------------------------------------------//
            //             Assemble scheme in the interior                 //
            //-------------------------------------------------------------//

            void assemble_scheme(Mat& A) override
            {
                if constexpr (cfg_t::scheme_type == SchemeType::NonLinear || cfg_t::scheme_type == SchemeType::LinearHeterogeneous)
                {
                    scheme().update_ghosts_if_needed(unknown());
                }

                // std::cout << "assemble_scheme() of " << this->name() << std::endl;
                // std::cout << "[" << mpi::communicator().rank() << "] assemble_scheme() of " << this->name() << std::endl;

                if (this->current_insert_mode() == INSERT_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(ADD_VALUES);
                }

                // if (mpi::communicator().rank() == 1)
                // {
                //     sleep(1);
                // }

                // Interior interfaces
                scheme().template for_each_interior_interface_and_coeffs<Run::Sequential, Get::Cells, /* include_periodic = */ false>(
                    unknown(),
                    [&](auto& interface_cells, auto& comput_cells, auto& left_cell_coeffs, auto& right_cell_coeffs)
                    {
                        bool left_cell_is_locally_owned  = is_locally_owned(interface_cells[0]);
                        bool right_cell_is_locally_owned = is_locally_owned(interface_cells[1]);
                        for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                        {
                            auto left_cell_row  = local_row_index(interface_cells[0], field_i);
                            auto right_cell_row = local_row_index(interface_cells[1], field_i);
                            for (unsigned int field_j = 0; field_j < input_n_comp; ++field_j)
                            {
                                for (std::size_t c = 0; c < stencil_size; ++c)
                                {
                                    double left_cell_coeff  = scheme().cell_coeff(left_cell_coeffs, c, field_i, field_j);
                                    double right_cell_coeff = scheme().cell_coeff(right_cell_coeffs, c, field_i, field_j);

                                    if constexpr (ghost_elimination_enabled)
                                    {
                                        auto it_ghost = this->m_ghost_recursion.find(comput_cells[c].index);
                                        if (it_ghost == this->m_ghost_recursion.end())
                                        {
                                            auto comput_cell_col = local_col_index(comput_cells[c], field_j);
                                            MatSetValueLocal(A, left_cell_row, comput_cell_col, left_cell_coeff, ADD_VALUES);
                                            MatSetValueLocal(A, right_cell_row, comput_cell_col, right_cell_coeff, ADD_VALUES);
                                        }
                                        else
                                        {
                                            auto& linear_comb = it_ghost->second;
                                            for (auto& [cell, coeff] : linear_comb)
                                            {
                                                auto comput_cell_col = local_col_index(static_cast<PetscInt>(cell), field_j);
                                                MatSetValueLocal(A, left_cell_row, comput_cell_col, left_cell_coeff * coeff, ADD_VALUES);
                                                MatSetValueLocal(A, right_cell_row, comput_cell_col, right_cell_coeff * coeff, ADD_VALUES);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        auto comput_cell_col = local_col_index(comput_cells[c], field_j);
                                        if (left_cell_is_locally_owned)
                                        {
                                            // if (mpi::communicator().rank() == 1)
                                            // {
                                            //     std::cout << "[" << mpi::communicator().rank() << "] Diff A[L" << left_cell_row << ", L"
                                            //               << comput_cell_col << "] = A[G" << global_row_index(interface_cells[0],
                                            //               field_i)
                                            //               << ", G" << global_col_index(comput_cells[c], field_j)
                                            //               << "] = " << left_cell_coeff << std::endl;
                                            // }
                                            MatSetValueLocal(A, left_cell_row, comput_cell_col, left_cell_coeff, ADD_VALUES);
                                        }
                                        if (right_cell_is_locally_owned)
                                        {
                                            // if (mpi::communicator().rank() == 1)
                                            // {
                                            //     std::cout << "[" << mpi::communicator().rank() << "] Diff A[L" << right_cell_row << ", L"
                                            //               << comput_cell_col << "] = A[G" << global_row_index(interface_cells[1],
                                            //               field_i)
                                            //               << ", G" << global_col_index(comput_cells[c], field_j)
                                            //               << "] = " << right_cell_coeff << std::endl;
                                            // }
                                            MatSetValueLocal(A, right_cell_row, comput_cell_col, right_cell_coeff, ADD_VALUES);
                                        }
                                    }
                                }
                            }

                            if (left_cell_is_locally_owned)
                            {
                                set_is_row_not_empty(left_cell_row);
                            }
                            if (right_cell_is_locally_owned)
                            {
                                set_is_row_not_empty(right_cell_row);
                            }
                        }
                    });

                // Boundary interfaces
                if (m_include_boundary_fluxes)
                {
                    scheme().for_each_boundary_interface_and_coeffs(
                        unknown(),
                        [&](auto& cell, auto& comput_cells, auto& coeffs)
                        {
                            if (!is_locally_owned(cell))
                            {
                                return;
                            }

                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
                                auto cell_row = local_row_index(cell, field_i);
                                for (unsigned int field_j = 0; field_j < input_n_comp; ++field_j)
                                {
                                    for (std::size_t c = 0; c < stencil_size; ++c)
                                    {
                                        double coeff         = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                        auto comput_cell_col = local_col_index(comput_cells[c], field_j);
                                        // if (mpi::communicator().rank() == 1)
                                        // {
                                        //     std::cout << "[" << mpi::communicator().rank() << "] boundary A[L" << cell_row << ", L"
                                        //               << comput_cell_col << "] = A[G" << global_row_index(cell, field_i) << ", G"
                                        //               << global_col_index(comput_cells[c], field_j) << "] = " << coeff << std::endl;
                                        // }
                                        MatSetValueLocal(A, cell_row, comput_cell_col, coeff, ADD_VALUES);
                                    }
                                }
                                set_is_row_not_empty(cell_row);
                            }
                        });
                }
            }
        };

    } // end namespace petsc
} // end namespace samurai
