#pragma once
#include "../../interface.hpp"
#include "../../schemes/fv/flux_based_scheme.hpp"
#include "FV_scheme_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <class Scheme>
        class Assembly<
            Scheme,
            std::enable_if_t<std::is_base_of_v<FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::bdry_cfg_t, typename Scheme::field_t>, Scheme>>>
            : public FVSchemeAssembly<Scheme>
        {
          protected:

            using base_class = FVSchemeAssembly<Scheme>;
            using base_class::cell_coeff;
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

            using scheme_t                                 = Scheme;
            using cfg_t                                    = typename Scheme::cfg_t;
            using bdry_cfg_t                               = typename Scheme::bdry_cfg;
            using field_t                                  = typename Scheme::field_t;
            static constexpr std::size_t output_field_size = cfg_t::output_field_size;
            static constexpr std::size_t stencil_size      = cfg_t::stencil_size;

            explicit Assembly(const Scheme& scheme)
                : base_class(scheme)
            {
                set_current_insert_mode(ADD_VALUES);
            }

            auto scheme_coefficients() const
            {
                return scheme().coefficients();
            }

          public:

            //-------------------------------------------------------------//
            //                     Sparsity pattern                        //
            //-------------------------------------------------------------//

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto scheme_coeffs_dir = scheme_coefficients()[d];
                    for_each_interior_interface(
                        mesh(),
                        scheme_coeffs_dir.flux.direction,
                        scheme_coeffs_dir.flux.stencil,
                        [&](auto& interface_cells, auto&)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += stencil_size * field_size;
                                    nnz[static_cast<std::size_t>(this->row_index(interface_cells[1], field_i))] += stencil_size * field_size;
                                }
                            }
                        });

                    for_each_boundary_interface(
                        mesh(),
                        scheme_coeffs_dir.flux.direction,
                        scheme_coeffs_dir.flux.stencil,
                        [&](auto& interface_cells, auto&)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += stencil_size * field_size;
                                }
                            }
                        });

                    auto opposite_direction = xt::eval(-scheme_coeffs_dir.flux.direction);
                    auto opposite_stencil   = xt::eval(-scheme_coeffs_dir.flux.stencil);
                    for_each_boundary_interface(
                        mesh(),
                        opposite_direction,
                        opposite_stencil,
                        [&](auto& interface_cells, auto&)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += stencil_size * field_size;
                                }
                            }
                        });
                }
            }

          public:

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

                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto scheme_coeffs_dir = scheme_coefficients()[d];
                    for_each_interior_interface(
                        mesh(),
                        scheme_coeffs_dir.flux.direction,
                        scheme_coeffs_dir.flux.stencil,
                        scheme_coeffs_dir.flux.get_flux_coeffs,
                        scheme_coeffs_dir.get_cell1_coeffs,
                        scheme_coeffs_dir.get_cell2_coeffs,
                        [&](auto& interface_cells, auto& comput_cells, auto& cell1_coeffs, auto& cell2_coeffs)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                auto interface_cell1_row = this->row_index(interface_cells[0], field_i);
                                auto interface_cell2_row = this->row_index(interface_cells[1], field_i);
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    for (std::size_t c = 0; c < stencil_size; ++c)
                                    {
                                        auto comput_cell_col = col_index(comput_cells[c], field_j);
                                        double cell1_coeff   = cell_coeff(cell1_coeffs, c, field_i, field_j);
                                        double cell2_coeff   = cell_coeff(cell2_coeffs, c, field_i, field_j);
                                        // if (cell1_coeff != 0)
                                        // {
                                        MatSetValue(A, interface_cell1_row, comput_cell_col, cell1_coeff, ADD_VALUES);
                                        // }
                                        // if (cell2_coeff != 0)
                                        // {
                                        MatSetValue(A, interface_cell2_row, comput_cell_col, cell2_coeff, ADD_VALUES);
                                        // }
                                        // MatSetValue(A, interface_cell1_row, interface_cell1_row, 0, ADD_VALUES);
                                        // MatSetValue(A, interface_cell2_row, interface_cell2_row, 0, ADD_VALUES);
                                    }
                                }
                                set_is_row_not_empty(interface_cell1_row);
                                set_is_row_not_empty(interface_cell2_row);
                            }
                        });

                    for_each_boundary_interface(mesh(),
                                                scheme_coeffs_dir.flux.direction,
                                                scheme_coeffs_dir.flux.stencil,
                                                scheme_coeffs_dir.flux.get_flux_coeffs,
                                                scheme_coeffs_dir.get_cell1_coeffs,
                                                [&](auto& interface_cells, auto& comput_cells, auto& coeffs)
                                                {
                                                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                    {
                                                        auto interface_cell0_row = this->row_index(interface_cells[0], field_i);
                                                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                        {
                                                            for (std::size_t c = 0; c < stencil_size; ++c)
                                                            {
                                                                double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                                                if (coeff != 0)
                                                                {
                                                                    auto comput_cell_col = col_index(comput_cells[c], field_j);
                                                                    MatSetValue(A, interface_cell0_row, comput_cell_col, coeff, ADD_VALUES);
                                                                }
                                                            }
                                                        }
                                                        set_is_row_not_empty(interface_cell0_row);
                                                    }
                                                });

                    auto opposite_direction             = xt::eval(-scheme_coeffs_dir.flux.direction);
                    Stencil<stencil_size, dim> reversed = xt::eval(xt::flip(scheme_coeffs_dir.flux.stencil, 0));
                    auto opposite_stencil               = xt::eval(-reversed);
                    for_each_boundary_interface(mesh(),
                                                opposite_direction,
                                                opposite_stencil,
                                                scheme_coeffs_dir.flux.get_flux_coeffs,
                                                scheme_coeffs_dir.get_cell2_coeffs,
                                                [&](auto& interface_cells, auto& comput_cells, auto& coeffs)
                                                {
                                                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                    {
                                                        auto interface_cell0_row = this->row_index(interface_cells[0], field_i);
                                                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                        {
                                                            for (std::size_t c = 0; c < stencil_size; ++c)
                                                            {
                                                                double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                                                if (coeff != 0)
                                                                {
                                                                    auto comput_cell_col = col_index(comput_cells[c], field_j);
                                                                    MatSetValue(A, interface_cell0_row, comput_cell_col, coeff, ADD_VALUES);
                                                                }
                                                            }
                                                        }
                                                        set_is_row_not_empty(interface_cell0_row);
                                                    }
                                                });
                }
            }
        };

    } // end namespace petsc
} // end namespace samurai
