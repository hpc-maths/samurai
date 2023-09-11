#pragma once
#include "../../interface.hpp"
#include "../../schemes/fv/flux_based_scheme.hpp"
#include "FV_scheme_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Assembly for linear and homogeneous schemes
         */
        template <class Scheme>
        class Assembly<Scheme, std::enable_if_t<is_FluxBasedScheme_v<Scheme> && Scheme::cfg_t::flux_type == FluxType::LinearHomogeneous>>
            : public FVSchemeAssembly<Scheme>
        {
          protected:

            using base_class = FVSchemeAssembly<Scheme>;
            using base_class::col_index;
            using base_class::dim;
            using base_class::field_size;
            using base_class::row_index;
            using base_class::set_is_row_not_empty;

          public:

            using base_class::ghost_elimination_enabled;
            using base_class::mesh;
            using base_class::scheme;
            using base_class::set_current_insert_mode;

            using scheme_t                                 = Scheme;
            using cfg_t                                    = typename Scheme::cfg_t;
            using bdry_cfg_t                               = typename Scheme::bdry_cfg;
            using field_t                                  = typename Scheme::field_t;
            using scheme_definition_t                      = typename Scheme::scheme_definition_t;
            using flux_stencil_coeffs_t                    = typename scheme_definition_t::flux_stencil_coeffs_t;
            static constexpr std::size_t output_field_size = cfg_t::output_field_size;
            static constexpr std::size_t stencil_size      = cfg_t::stencil_size;

            explicit Assembly(const Scheme& scheme)
                : base_class(scheme)
            {
                set_current_insert_mode(ADD_VALUES);
            }

            auto scheme_definition() const
            {
                return scheme().definition();
            }

            //-------------------------------------------------------------//
            //                     Sparsity pattern                        //
            //-------------------------------------------------------------//

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                auto definition = scheme_definition();
                for (std::size_t d = 0; d < dim; ++d)
                {
                    for_each_interior_interface(
                        mesh(),
                        definition[d].flux().direction,
                        definition[d].flux().stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    if constexpr (ghost_elimination_enabled)
                                    {
                                        for (std::size_t c = 0; c < stencil_size; ++c)
                                        {
                                            auto it_ghost = this->m_ghost_recursion.find(comput_cells[c].index);
                                            if (it_ghost == this->m_ghost_recursion.end())
                                            {
                                                nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += field_size;
                                                nnz[static_cast<std::size_t>(this->row_index(interface_cells[1], field_i))] += field_size;
                                            }
                                            else
                                            {
                                                auto& linear_comb = it_ghost->second;
                                                nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += linear_comb.size()
                                                                                                                             * field_size;
                                                nnz[static_cast<std::size_t>(this->row_index(interface_cells[1], field_i))] += linear_comb.size()
                                                                                                                             * field_size;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += stencil_size
                                                                                                                     * field_size;
                                        nnz[static_cast<std::size_t>(this->row_index(interface_cells[1], field_i))] += stencil_size
                                                                                                                     * field_size;
                                    }
                                }
                            }
                        });

                    for_each_boundary_interface(mesh(),
                                                definition[d].flux().direction,
                                                definition[d].flux().stencil,
                                                [&](auto& cell, auto&)
                                                {
                                                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                    {
                                                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                        {
                                                            nnz[static_cast<std::size_t>(this->row_index(cell, field_i))] += stencil_size
                                                                                                                           * field_size;
                                                        }
                                                    }
                                                });
                }
            }

            //-------------------------------------------------------------//
            //             Assemble scheme in the interior                 //
            //-------------------------------------------------------------//

            void assemble_scheme(Mat& A) override
            {
                // std::cout << "assemble_scheme() of " << this->name() << std::endl;

                if (this->current_insert_mode() == INSERT_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(ADD_VALUES);
                }

                // Interior interfaces
                scheme().for_each_interior_interface(
                    mesh(),
                    [&](auto& interface_cells, auto& comput_cells, auto& left_cell_coeffs, auto& right_cell_coeffs)
                    {
                        for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            auto left_cell_row  = this->row_index(interface_cells[0], field_i);
                            auto right_cell_row = this->row_index(interface_cells[1], field_i);
                            for (unsigned int field_j = 0; field_j < field_size; ++field_j)
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
                                            // if (left_cell_coeff != 0)
                                            // {
                                            MatSetValue(A, left_cell_row, comput_cell_col, left_cell_coeff, ADD_VALUES);
                                            // }
                                            // if (right_cell_coeff != 0)
                                            // {
                                            MatSetValue(A, right_cell_row, comput_cell_col, right_cell_coeff, ADD_VALUES);
                                            // }
                                            // MatSetValue(A, left_cell_row, left_cell_row, 0, ADD_VALUES);
                                            // MatSetValue(A, right_cell_row, right_cell_row, 0, ADD_VALUES);
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
                                        // if (left_cell_coeff != 0)
                                        // {
                                        MatSetValue(A, left_cell_row, comput_cell_col, left_cell_coeff, ADD_VALUES);
                                        // }
                                        // if (right_cell_coeff != 0)
                                        // {
                                        MatSetValue(A, right_cell_row, comput_cell_col, right_cell_coeff, ADD_VALUES);
                                        // }
                                        // MatSetValue(A, left_cell_row, left_cell_row, 0, ADD_VALUES);
                                        // MatSetValue(A, right_cell_row, right_cell_row, 0, ADD_VALUES);
                                    }
                                }
                            }
                            set_is_row_not_empty(left_cell_row);
                            set_is_row_not_empty(right_cell_row);
                        }
                    });

                // Boundary interfaces
                scheme().for_each_boundary_interface(mesh(),
                                                     [&](auto& cell, auto& comput_cells, auto& coeffs)
                                                     {
                                                         for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                         {
                                                             auto cell_row = this->row_index(cell, field_i);
                                                             for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                             {
                                                                 for (std::size_t c = 0; c < stencil_size; ++c)
                                                                 {
                                                                     double coeff = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                                                     if (coeff != 0)
                                                                     {
                                                                         auto comput_cell_col = col_index(comput_cells[c], field_j);
                                                                         MatSetValue(A, cell_row, comput_cell_col, coeff, ADD_VALUES);
                                                                     }
                                                                 }
                                                             }
                                                             set_is_row_not_empty(cell_row);
                                                         }
                                                     });
            }
        };

        /**
         * Assembly for linear and heterogeneous schemes
         */
        template <class Scheme>
        class Assembly<Scheme, std::enable_if_t<is_FluxBasedScheme_v<Scheme> && Scheme::cfg_t::flux_type == FluxType::LinearHeterogeneous>>
            : public FVSchemeAssembly<Scheme>
        {
        };

    } // end namespace petsc
} // end namespace samurai
