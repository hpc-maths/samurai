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
            using base_class::output_n_comp;
            using base_class::set_is_row_not_empty;

          public:

            using base_class::col_index;
            using base_class::ghost_elimination_enabled;
            using base_class::mesh;
            using base_class::row_index;
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

            // void sparsity_pattern(PetscInt& d_nz, PetscInt& o_nz) const override
            // {
            //     static constexpr PetscInt n_faces               = 2 * dim;
            //     static constexpr PetscInt half_stencil_per_face = (stencil_size % 2 == 0) ? stencil_size / 2 : (stencil_size + 1) / 2;
            //     static constexpr PetscInt number_of_neighbours  = n_faces * half_stencil_per_face;

            //     // d_nz = static_cast<PetscInt>((number_of_neighbours + 1) * input_n_comp);
            //     // o_nz = static_cast<PetscInt>(number_of_neighbours * input_n_comp);
            // }

            void sparsity_pattern_scheme(
#ifdef SAMURAI_WITH_MPI
                std::vector<PetscInt>& d_nnz,
                std::vector<PetscInt>& o_nnz
#else
                std::vector<PetscInt>& nnz
#endif
            ) const override
            {
                std::cout << "[" << mpi::communicator().rank() << "] sparsity_pattern_scheme() of interior interfaces" << std::endl;
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
                            int rank                  = mpi::communicator().rank();
                            bool cell_0_locally_owned = this->m_ownership[static_cast<std::size_t>(interface_cells[0].index)] == rank;
                            bool cell_1_locally_owned = this->m_ownership[static_cast<std::size_t>(interface_cells[1].index)] == rank;
#endif
                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
#if SAMURAI_WITH_MPI
                                std::size_t row_cell_0 = static_cast<std::size_t>(this->local_row_index(interface_cells[0], field_i));
                                std::size_t row_cell_1 = static_cast<std::size_t>(this->local_row_index(interface_cells[1], field_i));
#else
                                std::size_t row_cell_0 = static_cast<std::size_t>(this->row_index(interface_cells[0], field_i));
                                std::size_t row_cell_1 = static_cast<std::size_t>(this->row_index(interface_cells[1], field_i));
#endif
                            // if (mpi::communicator().rank() == 0)
                            // {
                            //     std::cout << fmt::format("rank {}: (nnz) interface between L{} and L{}\n",
                            //                              mpi::communicator().rank(),
                            //                              row_cell_0,
                            //                              row_cell_1);
                            // }

                            // for (unsigned int field_j = 0; field_j < input_n_comp; ++field_j)
                            // {
#ifdef SAMURAI_WITH_MPI
                                for (std::size_t c = 0; c < stencil_size; ++c)
                                {
                                    // if (mpi::communicator().rank() == 0 && (row_cell_0 == 19 || row_cell_1 == 19))
                                    // {
                                    //     std::cout << fmt::format("rank {}: comput_cells[{}] = L{}\n",
                                    //                              mpi::communicator().rank(),
                                    //                              c,
                                    //                              comput_cells[c].index);
                                    // }
                                    if (this->m_ownership[static_cast<std::size_t>(comput_cells[c].index)] == rank)
                                    {
                                        if (cell_0_locally_owned)
                                        {
                                            d_nnz.at(row_cell_0) += input_n_comp;
                                            std::cout
                                                << fmt::format("[{}]: d_nnz[L{}] = {} for col (index={}) L{} G{}\n",
                                                               mpi::communicator().rank(),
                                                               row_cell_0,
                                                               d_nnz[row_cell_0],
                                                               comput_cells[c].index,
                                                               this->m_local_cell_indices[static_cast<std::size_t>(comput_cells[c].index)],
                                                               this->m_global_cell_indices[static_cast<std::size_t>(comput_cells[c].index)]);
                                        }
                                        if (cell_1_locally_owned)
                                        {
                                            d_nnz.at(row_cell_1) += input_n_comp;
                                            std::cout
                                                << fmt::format("[{}]: d_nnz[L{}] = {} for col (index={}) L{} G{}\n",
                                                               mpi::communicator().rank(),
                                                               row_cell_1,
                                                               d_nnz[row_cell_1],
                                                               comput_cells[c].index,
                                                               this->m_local_cell_indices[static_cast<std::size_t>(comput_cells[c].index)],
                                                               this->m_global_cell_indices[static_cast<std::size_t>(comput_cells[c].index)]);
                                        }
                                        // if (mpi::communicator().rank() == 0 && row_cell_0 == 19)
                                        // {
                                        //     std::cout << fmt::format("rank {}: d_nnz[L{}] = {} for col L{}\n",
                                        //                              mpi::communicator().rank(),
                                        //                              row_cell_0,
                                        //                              d_nnz[row_cell_0],
                                        //                              comput_cells[c].index);
                                        // }
                                        // if (mpi::communicator().rank() == 0 && row_cell_1 == 19)
                                        // {
                                        //     std::cout << fmt::format("rank {}: d_nnz[L{}] = {} for col L{}\n",
                                        //                              mpi::communicator().rank(),
                                        //                              row_cell_1,
                                        //                              d_nnz[row_cell_1],
                                        //                              comput_cells[c].index);
                                        // }
                                    }
                                    else
                                    {
                                        if (cell_0_locally_owned)
                                        {
                                            o_nnz.at(row_cell_0) += input_n_comp;

                                            std::cout
                                                << fmt::format("[{}]: o_nnz[L{}] = {} for col (index={}) L{} G{}\n",
                                                               mpi::communicator().rank(),
                                                               row_cell_0,
                                                               o_nnz[row_cell_0],
                                                               comput_cells[c].index,
                                                               this->m_local_cell_indices[static_cast<std::size_t>(comput_cells[c].index)],
                                                               this->m_global_cell_indices[static_cast<std::size_t>(comput_cells[c].index)]);
                                        }
                                        if (cell_1_locally_owned)
                                        {
                                            o_nnz.at(row_cell_1) += input_n_comp;

                                            std::cout
                                                << fmt::format("[{}]: o_nnz[L{}] = {} for col (index={}) L{} G{}\n",
                                                               mpi::communicator().rank(),
                                                               row_cell_1,
                                                               o_nnz[row_cell_1],
                                                               comput_cells[c].index,
                                                               this->m_local_cell_indices[static_cast<std::size_t>(comput_cells[c].index)],
                                                               this->m_global_cell_indices[static_cast<std::size_t>(comput_cells[c].index)]);
                                        }
                                        // if (mpi::communicator().rank() == 0 && row_cell_0 == 19)
                                        // {
                                        //     std::cout << fmt::format("rank {}: o_nnz[L{}] = {}\n",
                                        //                              mpi::communicator().rank(),
                                        //                              row_cell_0,
                                        //                              o_nnz[row_cell_0]);
                                        // }
                                        // if (mpi::communicator().rank() == 0 && row_cell_1 == 19)
                                        // {
                                        //     std::cout << fmt::format("rank {}: o_nnz[L{}] = {}\n",
                                        //                              mpi::communicator().rank(),
                                        //                              row_cell_1,
                                        //                              o_nnz[row_cell_1]);
                                        // }
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
                                            nnz[row_cell_0] += static_cast<PetscInt>(input_n_comp);
                                            nnz[row_cell_1] += static_cast<PetscInt>(input_n_comp);
                                        }
                                        else
                                        {
                                            auto& linear_comb = it_ghost->second;
                                            nnz[row_cell_0] += static_cast<PetscInt>(linear_comb.size() * input_n_comp);
                                            nnz[row_cell_1] += static_cast<PetscInt>(linear_comb.size() * input_n_comp);
                                        }
                                    }
                                }
                                else
                                {
                                    nnz[row_cell_0] += static_cast<PetscInt>(stencil_size * input_n_comp);
                                    nnz[row_cell_1] += static_cast<PetscInt>(stencil_size * input_n_comp);
                                }
#endif
                                // }
                            }
                        });

                    if (m_include_boundary_fluxes)
                    {
                        std::cout << "[" << mpi::communicator().rank() << "] sparsity_pattern_scheme() of boundary interfaces" << std::endl;
                        for_each_boundary_interface__both_directions(
                            mesh(),
                            flux_def[d].direction,
                            flux_def[d].stencil,
                            [&](auto& cell, [[maybe_unused]] auto& comput_cells)
                            {
#ifdef SAMURAI_WITH_MPI
                                if (this->m_ownership[static_cast<std::size_t>(cell.index)] != mpi::communicator().rank())
                                {
                                    return;
                                }
#endif

                                for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                                {
#ifdef SAMURAI_WITH_MPI
                                    std::size_t row_cell = static_cast<std::size_t>(this->local_row_index(cell, field_i));
#else
                                    std::size_t row_cell = static_cast<std::size_t>(this->row_index(cell, field_i));
#endif
// for (unsigned int field_j = 0; field_j < input_n_comp; ++field_j)
// {
#ifdef SAMURAI_WITH_MPI
                                    for (std::size_t c = 0; c < stencil_size; ++c)
                                    {
                                        if (this->m_ownership[static_cast<std::size_t>(comput_cells[c].index)] == mpi::communicator().rank())
                                        {
                                            d_nnz.at(row_cell) += input_n_comp;
                                            // if (mpi::communicator().rank() == 1) // && row_cell == 19)
                                            // {
                                            //     std::cout << fmt::format("rank {}: d_nnz[L{}] = {}\n",
                                            //                              mpi::communicator().rank(),
                                            //                              row_cell,
                                            //                              d_nnz[row_cell]);
                                            // }
                                        }
                                        else
                                        {
                                            o_nnz.at(row_cell) += input_n_comp;
                                            // if (mpi::communicator().rank() == 1) // && row_cell == 19)
                                            // {
                                            //     std::cout << fmt::format("rank {}: o_nnz[L{}] = {}\n",
                                            //                              mpi::communicator().rank(),
                                            //                              row_cell,
                                            //                              o_nnz[row_cell]);
                                            // }
                                        }
                                    }
#else
                                    nnz[row_cell] += static_cast<PetscInt>(stencil_size * input_n_comp);
#endif
                                    // }
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
                // std::cout << "assemble_scheme() of " << this->name() << std::endl;

                // if (mpi::communicator().rank() == 1)
                // {
                //     sleep(10);
                // }

                std::cout << "[" << mpi::communicator().rank() << "] assemble_scheme()" << std::endl;

                if (this->current_insert_mode() == INSERT_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    std::cout << "[" << mpi::communicator().rank() << "] Flushing assembly to switch to ADD_VALUES mode\n";
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(ADD_VALUES);
                }

                // Interior interfaces
                scheme().template for_each_interior_interface_and_coeffs<Run::Sequential, Get::Cells, /* include_periodic = */ false>(
                    unknown(),
                    [&](auto& interface_cells, auto& comput_cells, auto& left_cell_coeffs, auto& right_cell_coeffs)
                    {
#ifdef SAMURAI_WITH_MPI
                        int rank                         = mpi::communicator().rank();
                        bool left_cell_is_locally_owned  = this->m_ownership[static_cast<std::size_t>(interface_cells[0].index)] == rank;
                        bool right_cell_is_locally_owned = this->m_ownership[static_cast<std::size_t>(interface_cells[1].index)] == rank;
#endif
                        for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                        {
#if SAMURAI_WITH_MPI
                            auto left_cell_local_row  = this->local_row_index(interface_cells[0], field_i);
                            auto right_cell_local_row = this->local_row_index(interface_cells[1], field_i);
                            auto left_cell_row        = this->global_row_index(interface_cells[0], field_i);
                            auto right_cell_row       = this->global_row_index(interface_cells[1], field_i);
#else
                            auto left_cell_row  = this->row_index(interface_cells[0], field_i);
                            auto right_cell_row = this->row_index(interface_cells[1], field_i);
#endif
                            // if (mpi::communicator().rank() == 1)
                            // {
                            //     std::cout << fmt::format("[{}]: (assemble) interface between L{} and L{}\n",
                            //                              mpi::communicator().rank(),
                            //                              left_cell_row,
                            //                              right_cell_row);
                            // }

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
                                            auto comput_cell_col = col_index(comput_cells[c], field_j);
                                            MatSetValue(A, left_cell_row, comput_cell_col, left_cell_coeff, ADD_VALUES);
                                            MatSetValue(A, right_cell_row, comput_cell_col, right_cell_coeff, ADD_VALUES);
                                        }
                                        else
                                        {
                                            auto& linear_comb = it_ghost->second;
                                            for (auto& [cell, coeff] : linear_comb)
                                            {
                                                auto comput_cell_col = col_index(static_cast<PetscInt>(cell), field_j);
                                                MatSetValue(A, left_cell_row, comput_cell_col, left_cell_coeff * coeff, ADD_VALUES);
                                                MatSetValue(A, right_cell_row, comput_cell_col, right_cell_coeff * coeff, ADD_VALUES);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        auto comput_cell_col = col_index(comput_cells[c], field_j);
#ifdef SAMURAI_WITH_MPI
                                        if (left_cell_is_locally_owned)
                                        {
                                            // if (mpi::communicator().rank() == 1) // && left_cell_row == 19)
                                            // {
                                            //     std::cout << fmt::format("[{}]: MatSetValue (left) to (G{}, G{}) : {}\n",
                                            //                              mpi::communicator().rank(),
                                            //                              left_cell_row,
                                            //                              comput_cell_col,
                                            //                              left_cell_coeff);
                                            // }
                                            MatSetValue(A, left_cell_row, comput_cell_col, left_cell_coeff, ADD_VALUES);
                                            // if (mpi::communicator().rank() == 1)
                                            // {
                                            //     std::cout << "  -> done\n";
                                            // }
                                        }
                                        if (right_cell_is_locally_owned)
                                        {
                                            // if (mpi::communicator().rank() == 1) // && right_cell_row == 19)
                                            // {
                                            // std::cout << fmt::format("[{}]: MatSetValue (right) to (G{}, G{}) : {}\n",
                                            //                          mpi::communicator().rank(),
                                            //                          right_cell_row,
                                            //                          comput_cell_col,
                                            //                          right_cell_coeff);
                                            // }
                                            MatSetValue(A, right_cell_row, comput_cell_col, right_cell_coeff, ADD_VALUES);
                                            // if (mpi::communicator().rank() == 1)
                                            // {
                                            //     std::cout << "  -> done\n";
                                            // }
                                        }
#else
                                        MatSetValue(A, left_cell_row, comput_cell_col, left_cell_coeff, ADD_VALUES);
                                        MatSetValue(A, right_cell_row, comput_cell_col, right_cell_coeff, ADD_VALUES);
#endif
                                    }
                                }
                            }

#ifdef SAMURAI_WITH_MPI
                            if (left_cell_is_locally_owned)
                            {
                                set_is_row_not_empty(left_cell_local_row);
                            }
                            if (right_cell_is_locally_owned)
                            {
                                set_is_row_not_empty(right_cell_local_row);
                            }
#else
                            set_is_row_not_empty(left_cell_row);
                            set_is_row_not_empty(right_cell_row);
#endif
                        }
                    });

                // Boundary interfaces
                if (m_include_boundary_fluxes)
                {
                    scheme().for_each_boundary_interface_and_coeffs(
                        unknown(),
                        [&](auto& cell, auto& comput_cells, auto& coeffs)
                        {
#ifdef SAMURAI_WITH_MPI
                            if (this->m_ownership[static_cast<std::size_t>(cell.index)] != mpi::communicator().rank())
                            {
                                return;
                            }
#endif
                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
#if SAMURAI_WITH_MPI
                                auto cell_row     = this->global_row_index(cell, field_i);
                                auto cell_loc_row = this->local_row_index(cell, field_i);
#else
                                auto cell_row = this->row_index(cell, field_i);
#endif
                                for (unsigned int field_j = 0; field_j < input_n_comp; ++field_j)
                                {
                                    for (std::size_t c = 0; c < stencil_size; ++c)
                                    {
                                        double coeff         = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                        auto comput_cell_col = col_index(comput_cells[c], field_j);

                                        // if (mpi::communicator().rank() == 1) // && right_cell_row == 19)
                                        // {
                                        // std::cout << fmt::format("[{}]: MatSetValue (boundary) to (G{}, G{}) : {}\n",
                                        //                          mpi::communicator().rank(),
                                        //                          cell_row,
                                        //                          comput_cell_col,
                                        //                          coeff);
                                        // }
                                        MatSetValue(A, cell_row, comput_cell_col, coeff, ADD_VALUES);
                                    }
                                }
#ifdef SAMURAI_WITH_MPI
                                set_is_row_not_empty(cell_loc_row);
#else
                                set_is_row_not_empty(cell_row);
#endif
                            }
                        });
                }
            }
        };

    } // end namespace petsc
} // end namespace samurai
